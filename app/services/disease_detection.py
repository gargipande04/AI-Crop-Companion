"""
disease_detection.py — Plant Disease Detection Engine

Loads a ResNet-18 model trained on the Diseases Dataset and exposes:
  - load_disease_model()      : load weights into memory (called at startup)
  - preprocess_image()        : resize / normalise image bytes into a model-ready tensor
  - run_disease_inference()   : synchronous inference, returns structured result dict
  - diagnose()                : async FastAPI endpoint handler (offloads to executor)
  - disease_model_status()    : lightweight status dict for /disease-model-status
"""

import io
import json

from fastapi import File, HTTPException, UploadFile

from app.core.config import MODELS_DIR

TREATMENTS = {
    "Black_knot": {
        "display": "Black Knot",
        "crop": "Any (primarily stone fruits — cherry, plum, apricot)",
        "severity": "High",
        "description": "Fungal disease (Apiosporina morbosa) causing hard, black, elongated galls on branches. Girdles and kills branches over time; can kill whole trees if untreated.",
        "organic": [
            "Prune infected branches at least 10–15 cm below the visible gall during dormancy (late winter)",
            "Immediately bag and destroy all pruned material — do not compost",
            "Apply copper-based fungicide (Bordeaux mixture) at bud swell, then at 2-week intervals through bloom",
            "Remove any wild cherry or plum trees nearby — they are reservoir hosts",
        ],
        "chemical": [
            "Thiophanate-methyl (Topsin-M) — apply at green tip, pink bud, and petal fall",
            "Captan fungicide as a protective spray every 10–14 days during wet spring weather",
        ],
        "prevention": [
            "Plant resistant varieties — some cherry cultivars show tolerance",
            "Inspect trees every spring and prune out galls before they mature and release spores",
            "Maintain tree vigour — stressed trees are far more susceptible",
        ],
    },
    "Chlorosis": {
        "display": "Chlorosis (Nutrient Deficiency)",
        "crop": "Any plant",
        "severity": "Moderate",
        "description": "Yellowing of leaves while veins remain green. Not an infection — caused by iron, manganese, or magnesium deficiency, usually triggered by high soil pH (above 7.0) locking nutrients out of plant-available forms.",
        "organic": [
            "Apply chelated iron foliar spray — fastest visible response within days",
            "Work composted organic matter into soil to lower pH gradually",
            "Mulch with pine bark or acidic compost around the root zone",
            "Coffee grounds worked into soil add acidity slowly",
        ],
        "chemical": [
            "Ferrous sulfate soil drench — acidifies soil and adds iron simultaneously",
            "Soil acidifier (elemental sulfur) at 100–200 g/m² — retest pH after 8 weeks",
            "Manganese sulfate or magnesium sulfate (Epsom salt) foliar spray if those specific deficiencies are confirmed by a soil test",
        ],
        "prevention": [
            "Test soil pH before planting — target 6.0–6.5 for most plants",
            "Avoid over-irrigation which raises effective pH and leaches nutrients",
            "Never plant acid-loving plants (blueberries, azaleas) in alkaline soil without amendment",
        ],
    },
    "Dog_vomit_slime_mold": {
        "display": "Dog Vomit Slime Mould",
        "crop": "Any (affects mulch and soil surface, not the plant itself)",
        "severity": "Low",
        "description": "Fuligo septica — a saprophytic slime mould that appears as bright yellow or orange foam on mulch or soil after wet weather. It feeds on bacteria and decaying matter, not on plants. Harmless to the plant but alarming to look at.",
        "organic": [
            "Simply rake or break it up and let it dry — it disappears within days on its own",
            "If on a lawn, remove with a shovel and dispose of it",
            "No treatment necessary — it is not a plant pathogen",
        ],
        "chemical": [
            "No chemical treatment needed or recommended — slime moulds are beneficial decomposers",
        ],
        "prevention": [
            "Improve drainage around mulched areas — slime moulds thrive in persistently wet conditions",
            "Avoid piling mulch too deeply (keep below 7 cm) to improve air circulation",
            "Turn mulch occasionally to reduce moisture buildup",
        ],
    },
    "Elderberry_rust": {
        "display": "Elderberry Rust",
        "crop": "Elderberry (Sambucus spp.)",
        "severity": "Moderate",
        "description": "Fungal rust disease (Puccinia sambuci) causing bright orange-yellow powdery pustules on leaves and stems. Requires two hosts to complete its life cycle — elderberry and sedges (Carex spp.).",
        "organic": [
            "Remove and destroy infected leaves promptly — do not compost",
            "Apply sulfur-based fungicide spray at first sign of orange pustules",
            "Eliminate sedge grasses (Carex spp.) near elderberry plants to break the disease cycle",
        ],
        "chemical": [
            "Myclobutanil or propiconazole — effective triazole fungicides against rusts",
            "Apply at 10–14 day intervals during wet weather when rust pressure is high",
        ],
        "prevention": [
            "Create distance between elderberry plantings and any sedge-rich wetland areas",
            "Choose rust-resistant elderberry cultivars where available",
            "Ensure good airflow through pruning to reduce humidity around foliage",
        ],
    },
    "Golden_canker": {
        "display": "Golden Canker",
        "crop": "Dogwood (Cornus spp.)",
        "severity": "High",
        "description": "Fungal canker disease (Cryptodiaporthe corni) causing bright golden-yellow patches on bark, sunken dead areas, and dieback of branches. Enters through wounds and spreads in wet conditions.",
        "organic": [
            "Prune infected branches 15–20 cm below visible canker edge during dry weather",
            "Sterilise pruning tools with 70% isopropyl alcohol between cuts",
            "Apply copper fungicide paste to pruning wounds immediately after cutting",
            "Remove severely infected trees entirely to prevent spread to nearby dogwoods",
        ],
        "chemical": [
            "Thiophanate-methyl or copper-based fungicide applied as a bark spray during dormancy",
            "Repeat applications in spring at bud break and again after petal fall",
        ],
        "prevention": [
            "Avoid wounding bark during mowing or strimming — wounds are the primary entry point",
            "Plant dogwoods in well-drained soil — waterlogged roots increase susceptibility dramatically",
            "Choose blight-resistant species such as Cornus kousa (Japanese dogwood) over native C. florida",
        ],
    },
    "Gymnosporangium_Rusts": {
        "display": "Gymnosporangium Rust (Cedar-Apple / Hawthorn Rust)",
        "crop": "Rosaceous plants — apple, pear, hawthorn, quince (alternate host: juniper/cedar)",
        "severity": "Moderate",
        "description": "Complex rust fungi (Gymnosporangium spp.) requiring two different host plants to complete their life cycle: a rosaceous plant (apple, hawthorn) and a juniper or cedar. Causes bright orange-red spots on leaves and fruit of the rosaceous host; jelly-like orange galls on juniper in spring.",
        "organic": [
            "Remove juniper/cedar trees within 1–2 km if possible — they are the alternate host",
            "Apply sulfur-based fungicide from pink bud stage through 3 weeks post-petal fall",
            "Remove and destroy galls from juniper in late winter before they release spores",
        ],
        "chemical": [
            "Myclobutanil (Rally) — most effective, apply at bud break and every 7–10 days through bloom",
            "Propiconazole or tebuconazole as alternatives",
            "Timing is critical — apply before infection, not after visible spots appear",
        ],
        "prevention": [
            "Plant rust-resistant apple and crabapple varieties (Liberty, Redfree, William's Pride)",
            "Never plant junipers and apples or hawthorns near each other",
            "Monitor juniper for orange gelatinous galls in early spring — remove immediately",
        ],
    },
    "peach_leaf_curl": {
        "display": "Peach Leaf Curl",
        "crop": "Peach, nectarine (Prunus persica)",
        "severity": "High",
        "description": "Fungal disease (Taphrina deformans) causing leaves to pucker, thicken, and turn red/pink/yellow as they emerge in spring. Severely infected leaves drop early, weakening the tree and reducing fruit yield. One of the most common and damaging peach diseases worldwide.",
        "organic": [
            "Apply copper-based fungicide (Bordeaux mixture or copper hydroxide) in autumn after leaf drop AND again at late dormancy just before buds swell — timing is everything",
            "A single well-timed copper spray prevents infection for the entire season",
            "Remove and destroy fallen infected leaves",
        ],
        "chemical": [
            "Copper oxychloride or copper hydroxide — applied once in autumn at leaf fall, highly effective",
            "Ziram or chlorothalonil as alternatives if copper resistance is suspected",
            "Note: in-season sprays after symptoms appear have NO effect — the fungus is already inside the leaf",
        ],
        "prevention": [
            "The single most important action: spray copper every autumn without fail — even one missed year causes severe infection the following spring",
            "Plant resistant varieties — Frost, Redhaven, and Indian Free show good resistance",
            "Rain covers over trees during dormancy prevent spore germination on buds",
        ],
    },
    "Powdery_Mildew": {
        "display": "Powdery Mildew",
        "crop": "Any plant (extremely wide host range)",
        "severity": "Moderate",
        "description": "White powdery fungal coating (various Erysiphales species) on leaf surfaces, stems, and buds. Unlike most fungi, thrives in warm dry days with cool humid nights — does NOT need wet leaves to spread. Weakens plants and reduces yield but rarely kills.",
        "organic": [
            "Diluted milk spray (40% fresh milk, 60% water) applied weekly — scientifically proven effective",
            "Potassium bicarbonate spray (5 g per litre) — changes surface pH, kills spores on contact",
            "Neem oil (2%) spray every 7–10 days",
            "Remove and bin heavily infected leaves",
        ],
        "chemical": [
            "Myclobutanil or trifloxystrobin — systemic fungicides, move inside plant tissue",
            "Sulfur-based fungicide — very effective but avoid applying above 32°C (causes leaf burn)",
            "Tebuconazole — particularly effective on cereals and vegetables",
        ],
        "prevention": [
            "Space plants widely — good airflow is the single best preventative measure",
            "Avoid excess nitrogen fertiliser — lush soft growth is most susceptible",
            "Plant resistant varieties wherever available — most modern cultivars have mildew resistance bred in",
            "Water soil not leaves; avoid evening watering",
        ],
    },
    "Sooty_Mold": {
        "display": "Sooty Mould",
        "crop": "Any plant",
        "severity": "Low",
        "description": "Black sooty coating on leaves caused by fungi (Capnodium, Fumago spp.) growing on honeydew secreted by sap-sucking insects — aphids, scale, whitefly, or mealybugs. The mould itself does not infect the plant, but it blocks sunlight and indicates an active insect infestation underneath.",
        "organic": [
            "Identify and control the insect producing the honeydew — the mould disappears once the insects are gone",
            "Spray with strong water jet to dislodge aphids or scale from undersides of leaves",
            "Apply neem oil or insecticidal soap to kill the insect vector",
            "Wipe mould off leaves with a damp cloth — it comes off easily once dry",
        ],
        "chemical": [
            "Treat the insect cause, not the mould: imidacloprid systemic insecticide for scale/aphids on ornamentals",
            "Horticultural oil spray smothers scale insects and loosens sooty mould simultaneously",
        ],
        "prevention": [
            "Inspect plants weekly for early aphid or scale infestations — sooty mould is always a secondary symptom",
            "Encourage natural predators: ladybirds, lacewings, parasitic wasps eat the aphids that cause honeydew",
            "Avoid over-fertilising with nitrogen — it produces the lush new growth aphids prefer",
        ],
    },
    "Tar_Spot": {
        "display": "Tar Spot",
        "crop": "Maple (Acer spp.) primarily; also some grasses",
        "severity": "Low",
        "description": "Fungal disease (Rhytisma acerinum) producing distinctive raised, shiny black tar-like spots on maple leaves. Looks alarming but causes minimal real damage — mainly cosmetic. Infected leaves drop early but trees recover fully each season.",
        "organic": [
            "Rake and destroy all fallen leaves in autumn — the fungus overwinters in leaf debris and reinfects the following spring",
            "Do not compost infected leaves — bag them for landfill",
            "No in-season sprays are necessary or effective once spots are visible",
        ],
        "chemical": [
            "Preventative copper or chlorothalonil spray at bud break in spring — only worthwhile if cosmetic appearance is critical (e.g. specimen trees)",
            "In-season treatment after spots appear has no benefit",
        ],
        "prevention": [
            "Annual leaf clearance in autumn is the only reliably effective control measure",
            "Improve air circulation by thinning the canopy",
            "Healthy well-fed trees tolerate tar spot with no significant yield or vigour loss",
        ],
    },
}

# Maps severity labels to hex colours used by the frontend severity badge.
SEVERITY_COLOUR = {
    "None": "#2e5e35",
    "Low": "#4a8c52",
    "Moderate": "#c4a84a",
    "High": "#b5471d",
    "Critical": "#8b0000",
}

# Module-level model state — populated by load_disease_model() at startup.
_disease_model = None        # ResNet-18 nn.Module in eval mode
_disease_classes = None      # List[str] mapping output index → class name
_disease_model_meta = {"loaded": False, "val_acc": None, "n_classes": 0}


def load_disease_model() -> bool:
    """Load the disease ResNet-18 checkpoint into memory.

    Reads disease_model.pth and class_names.json from MODELS_DIR. Populates
    the module-level globals so subsequent calls to run_disease_inference()
    can run without re-loading.

    Returns True on success, False if the model file is missing or corrupt.
    torch is imported inside this function to avoid a LightGBM+PyTorch
    segfault when both are imported at module level on macOS.
    """
    global _disease_model, _disease_classes, _disease_model_meta

    model_path = MODELS_DIR / "disease_model.pth"
    class_path = MODELS_DIR / "class_names.json"

    print(f"Looking for disease model at: {model_path}")

    if not model_path.exists():
        print("   Not found — run: python train_disease_model.py --data ./Diseases_Dataset")
        return False
    if not class_path.exists():
        print(f"   class_names.json not found at {class_path}")
        return False

    try:
        # Deferred imports — importing torch at module level alongside LightGBM
        # causes a segfault on macOS due to conflicting OpenMP runtimes.
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models

        checkpoint = torch.load(model_path, map_location="cpu")
        n_classes = checkpoint["n_classes"]

        # Load the ordered list of class names saved during training.
        with open(class_path) as f:
            _disease_classes = json.load(f)

        # Reconstruct the same architecture used in train_disease_model.py:
        # standard ResNet-18 with the final FC layer replaced by Dropout + Linear.
        net = tv_models.resnet18(weights=None)
        net.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(net.fc.in_features, n_classes),
        )
        net.load_state_dict(checkpoint["model_state"])
        net.eval()  # disable dropout and batch-norm training behaviour

        _disease_model = net
        _disease_model_meta = {
            "loaded": True,
            "val_acc": checkpoint.get("val_acc"),
            "n_classes": n_classes,
            "epoch": checkpoint.get("epoch"),
        }
        print(f"Disease model loaded — val_acc={checkpoint.get('val_acc', '?'):.4f}, {n_classes} classes")
        return True

    except Exception as e:
        print(f"Failed to load disease model: {e}")
        return False


def preprocess_image(image_bytes: bytes):
    """Convert raw image bytes into a normalised (1, 3, 224, 224) tensor.

    Applies the standard ImageNet preprocessing pipeline:
    resize to 256 → centre-crop to 224 → ToTensor → Normalize.
    Also used by pest_detection.py for pest model inference.
    """
    from PIL import Image
    from torchvision import transforms

    tfm = transforms.Compose(
        [
            transforms.Resize(256),          # scale shorter edge to 256px
            transforms.CenterCrop(224),      # crop to the 224×224 expected by ResNet
            transforms.ToTensor(),           # HWC uint8 → CHW float32 in [0, 1]
            transforms.Normalize(            # ImageNet mean/std normalisation
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ]
    )
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return tfm(img).unsqueeze(0)  # add batch dimension → (1, 3, 224, 224)


def run_disease_inference(image_bytes: bytes) -> dict:
    """Run synchronous disease classification on image bytes.

    Lazily loads the model if it has not been loaded yet (e.g. if startup
    loading failed). Returns a dict with prediction, confidence, top-3
    alternatives, treatment advice, and model metadata. Raises HTTP 503 if
    the model file is missing.

    Called inside run_in_executor so it does not block the async event loop.
    """
    import torch
    import torch.nn.functional as F

    # Lazy-load if startup loading was skipped or failed.
    if not _disease_model_meta["loaded"]:
        ok = load_disease_model()
        if not ok:
            raise HTTPException(
                status_code=503,
                detail="Disease model not trained yet. Run: python train_disease_model.py --data ./Diseases_Dataset",
            )

    tensor = preprocess_image(image_bytes)

    # Run forward pass with gradient tracking disabled — saves memory and speeds up inference.
    with torch.no_grad():
        logits = _disease_model(tensor)
        probs = F.softmax(logits, dim=1)[0]  # convert raw scores to probabilities

    # Pull top-3 predictions; guard against models with fewer than 3 classes.
    top3_vals, top3_idx = torch.topk(probs, k=min(3, len(_disease_classes)))
    top_class = _disease_classes[top3_idx[0].item()]
    top_confidence = round(float(top3_vals[0]) * 100, 1)

    # Build the runner-up list (indices 1 and 2) shown in the UI as alternatives.
    alternatives = [
        {
            "class": _disease_classes[idx.item()],
            "display": TREATMENTS.get(_disease_classes[idx.item()], {}).get(
                "display",
                _disease_classes[idx.item()].replace("_", " "),
            ),
            "confidence": round(float(val) * 100, 1),
        }
        for val, idx in zip(top3_vals[1:], top3_idx[1:])
    ]

    # Look up treatment advice; fall back gracefully if the class has no entry.
    treatment = TREATMENTS.get(
        top_class,
        {
            "display": top_class.replace("_", " "),
            "crop": "Unknown",
            "severity": "Unknown",
            "description": "No specific treatment data available for this class.",
            "organic": [],
            "chemical": [],
            "prevention": [],
        },
    )

    return {
        "predicted_class": top_class,
        "confidence": top_confidence,
        "display_name": treatment["display"],
        "crop": treatment["crop"],
        "severity": treatment["severity"],
        "severity_colour": SEVERITY_COLOUR.get(treatment["severity"], "#888"),  # hex for UI badge
        "description": treatment["description"],
        "treatment": {
            "organic": treatment["organic"],
            "chemical": treatment["chemical"],
            "prevention": treatment["prevention"],
        },
        "alternatives": alternatives,
        "model_accuracy": _disease_model_meta,
    }


async def diagnose(file: UploadFile = File(...)):
    """FastAPI endpoint: classify a plant disease from an uploaded image.

    Reads the uploaded file, offloads inference to a thread-pool executor to
    keep the async event loop free, and returns the structured result dict.
    """
    import asyncio

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file received.")

    loop = asyncio.get_event_loop()
    try:
        # run_in_executor moves the blocking torch call off the async event loop.
        result = await loop.run_in_executor(None, run_disease_inference, image_bytes)
    except HTTPException:
        raise  # pass through 503 (model not found) unchanged
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Diagnosis error: {str(e)}")

    return result


def disease_model_status():
    """Return the current model metadata dict (loaded flag, val_acc, n_classes)."""
    return _disease_model_meta
