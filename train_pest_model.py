"""
Farm Insect Detector — Training Script
Dataset : farm_insects (15 classes)
            Africanized Honey Bees (Killer Bees), Aphids, Armyworms,
            Brown Marmorated Stink Bugs, Cabbage Loopers, Citrus Canker,
            Colorado Potato Beetles, Corn Borers, Corn Earworms,
            Fall Armyworms, Fruit Flies, Spider Mites, Thrips,
            Tomato Hornworms, Western Corn Rootworms

Model   : ResNet-18 with transfer learning
Output  : models/pest_model.pth  +  models/pest_class_names.json

Run once before starting main.py:
    python train_pest_model.py --data "/Users/gargipande/Downloads/farm_insects" --epochs 20
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--data",   default="./farm_insects", help="Path to dataset root")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch",  type=int, default=32)
parser.add_argument("--lr",     type=float, default=1e-3)
parser.add_argument("--out",    default="./models", help="Where to save model + class names")
args = parser.parse_args()

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if available, else CPU
DATA_DIR  = Path(args.data)
OUT_DIR   = Path(args.out)
MODEL_OUT = OUT_DIR / "pest_model.pth"          # checkpoint loaded by pest_detection.py
CLASS_OUT = OUT_DIR / "pest_class_names.json"   # class index → name mapping
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device : {DEVICE}")
print(f"Data   : {DATA_DIR}")
print(f"Epochs : {args.epochs}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training augmentations — random crops/flips/jitter reduce overfitting on a small dataset.
train_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),           # random crop vs. centre crop adds positional variety
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # lighting variation
    transforms.RandomRotation(15),        # ±15° rotation mimics different camera angles
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Validation transforms — deterministic, matches inference preprocessing in pest_detection.py.
val_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ImageFolder expects subdirectory-per-class structure: DATA_DIR/ClassName/img.jpg
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfm)
class_names  = full_dataset.classes   # alphabetically sorted list of class names
n_classes    = len(class_names)

# 80/20 random train/val split with a fixed seed for reproducibility.
n_val   = int(len(full_dataset) * 0.2)
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
# Replace the val subset's dataset reference so it uses val_tfm (no augmentation).
val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_tfm)

# num_workers=0 avoids multiprocessing issues on macOS; pin_memory only helps with CUDA.
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=0, pin_memory=False)

print(f"Classes : {n_classes}  |  Train : {n_train:,}  |  Val : {n_val:,}")
print(f"Class names: {class_names}")

# Load ResNet-18 pretrained on ImageNet — reusing learned feature detectors
# dramatically reduces the training data needed for good accuracy.
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers first — only the new head will train in early epochs.
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 (the deepest conv block) so it can adapt to pest-specific features.
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace the default 1000-class head with a Dropout + Linear head for n_classes.
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, n_classes),
)
model = model.to(DEVICE)

# Adam with weight decay — only updates the unfrozen parameters (layer4 + fc).
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=1e-4,
)
# Decay LR by 10× every 7 epochs to refine weights as training progresses.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0  # track best so we only save improved checkpoints

for epoch in range(1, args.epochs + 1):
    # Training phase 
    model.train()  # enables dropout and batch-norm training mode
    train_loss = train_correct = train_total = 0
    t0 = time.time()

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate weighted loss (loss.item() is the mean over the batch).
        train_loss    += loss.item() * imgs.size(0)
        preds          = outputs.argmax(1)
        train_correct += (preds == labels).sum().item()
        train_total   += imgs.size(0)

    train_acc  = train_correct / train_total
    train_loss = train_loss    / train_total

    # Validation phase 
    model.eval()  # disables dropout; batch-norm uses running stats
    val_correct = val_total = 0
    with torch.no_grad():  # no gradients needed — saves memory
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

    # Save checkpoint only when val_acc improves — keeps the best weights, not the last.
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_acc":     val_acc,
            "n_classes":   n_classes,  # saved so load_pest_model() can rebuild the head
        }, MODEL_OUT)
        print(f"  Saved best model (val_acc={val_acc:.4f})")

    scheduler.step()  # step LR scheduler after each epoch

# Save the class name list so pest_detection.py can map output indices to names.
with open(CLASS_OUT, "w") as f:
    json.dump(class_names, f, indent=2)

print(f"\nDone. Best val accuracy : {best_val_acc:.4f}")
print(f"Model saved   -> {MODEL_OUT}")
print(f"Classes saved -> {CLASS_OUT}")
