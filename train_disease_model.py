"""
Plant Disease Detector — Training Script
Dataset : Chakraborty et al. "Diseases Dataset" (Kaggle)
          https://www.kaggle.com/datasets/rejoychakraborty/plant-disease-dataset
          10 disease classes, real field photos, plant-invariant:
            Black_knot, Chlorosis, Dog_vomit_slime_mold, Elderberry_rust,
            Golden_canker, Gymnosporangium_Rusts, peach_leaf_curl,
            Powdery_Mildew, Sooty_Mold, Tar_Spot

Model   : ResNet-18 with transfer learning (fine-tune last block + classifier)
Output  : models/disease_model.pth  +  models/class_names.json

Run once before starting main.py:
    pip install torch torchvision pillow
    python train_disease_model.py --data ./Diseases_Dataset --epochs 15

On a GPU  : ~5 minutes   (small dataset — fast)
On CPU    : ~45 minutes

Directory layout expected (ImageFolder format):
    Diseases_Dataset/
        Black_knot/
            img001.jpg  ...
        Chlorosis/
        Dog_vomit_slime_mold/
        Elderberry_rust/
        Golden_canker/
        Gymnosporangium_Rusts/
        peach_leaf_curl/
        Powdery_Mildew/
        Sooty_Mold/
        Tar_Spot/
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

#  CLI args 
parser = argparse.ArgumentParser()
parser.add_argument("--data",   default="./Diseases_Dataset", help="Path to dataset root")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch",  type=int, default=32)
parser.add_argument("--lr",     type=float, default=1e-3)
parser.add_argument("--out",    default="./models",  help="Where to save model + class names")
args = parser.parse_args()

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR  = Path(args.data)
OUT_DIR   = Path(args.out)
MODEL_OUT = OUT_DIR / "disease_model.pth"
CLASS_OUT = OUT_DIR / "class_names.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device : {DEVICE}")
print(f"Data   : {DATA_DIR}")
print(f"Epochs : {args.epochs}")

#  Transforms 
# ImageNet mean/std because ResNet was pre-trained on ImageNet.
# Training transform adds augmentations to improve generalisation —
# random flips and colour jitter mean the model sees each image in
# many slightly different forms, making it robust to photo angle and
# lighting conditions in real farmer uploads.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

#  Dataset 
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfm)
class_names  = full_dataset.classes          # e.g. ["Apple___Apple_scab", ...]
n_classes    = len(class_names)

# 80/20 split (deterministic)
n_val   = int(len(full_dataset) * 0.2)
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
# Validation uses the clean (no-augmentation) transform
val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_tfm)

train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=0, pin_memory=False)

print(f"Classes : {n_classes}  |  Train : {n_train:,}  |  Val : {n_val:,}")

#  Model 
# ResNet-18 pre-trained on ImageNet.
# Transfer learning strategy: freeze everything except the last
# residual block (layer4) and the final classifier.
# This way the early layers (which detect edges, textures, colours —
# universal visual features) stay intact from ImageNet training.
# Only the high-level disease-specific features are re-learned.
# This needs far fewer epochs and far less data than training from scratch.

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last residual block so it adapts to plant imagery
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace the final FC layer: 512 → n_classes
# Dropout(0.4) reduces overfitting — randomly zeros 40 % of neurons
# during each training step, forcing the network not to rely on any
# single neuron.
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, n_classes),
)
model = model.to(DEVICE)

#  Optimiser & scheduler 
# Only optimise parameters that require gradients (layer4 + fc)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=1e-4,    # L2 regularisation — penalises large weights
)

# Reduce LR by 10× every 7 epochs if val loss stops improving.
# This lets the model make big updates early and fine-grained ones later.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

#  Training loop 
best_val_acc = 0.0

for epoch in range(1, args.epochs + 1):
    # — Train —
    model.train()
    train_loss = train_correct = train_total = 0
    t0 = time.time()

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * imgs.size(0)
        preds          = outputs.argmax(1)
        train_correct += (preds == labels).sum().item()
        train_total   += imgs.size(0)

    train_acc  = train_correct / train_total
    train_loss = train_loss    / train_total

    # — Validate —
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds         = model(imgs).argmax(1)
            val_correct  += (preds == labels).sum().item()
            val_total    += imgs.size(0)

    val_acc = val_correct / val_total
    elapsed = time.time() - t0

    print(f"Epoch {epoch:02d}/{args.epochs}  "
          f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
          f"loss={train_loss:.4f}  ({elapsed:.0f}s)")

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_acc":     val_acc,
            "n_classes":   n_classes,
        }, MODEL_OUT)
        print(f"   Saved best model (val_acc={val_acc:.4f})")

    scheduler.step()

#  Save class names 
with open(CLASS_OUT, "w") as f:
    json.dump(class_names, f, indent=2)

print(f"\nDone. Best val accuracy : {best_val_acc:.4f}")
print(f"Model saved  → {MODEL_OUT}")
print(f"Classes saved→ {CLASS_OUT}")
