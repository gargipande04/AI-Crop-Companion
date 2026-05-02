"""
pest_detection.py — Farm Pest Identification Engine

Loads a ResNet-18 model trained on the farm insects dataset and exposes:
  - load_pest_model()       : load weights into memory (called at startup)
  - run_pest_inference()    : synchronous inference, returns structured result dict
  - identify_pest()         : async FastAPI endpoint handler (offloads to executor)
  - pest_model_status()     : lightweight status dict for /pest-model-status

Image preprocessing is shared with disease_detection via preprocess_image().
"""

import json

from fastapi import File, HTTPException, UploadFile

from app.core.config import MODELS_DIR
from app.services.disease_detection import SEVERITY_COLOUR, preprocess_image

PEST_INFO = {
    "Africanized Honey Bees (Killer Bees)": {
        "display": "Africanized Honey Bees (Killer Bees)",
        "crops": "All field crops — threat is to farm workers, not plants directly",
        "severity": "Critical",
        "description": "Highly aggressive hybrid bees that swarm and sting in large numbers when disturbed. A serious human safety hazard on farms. Colonies establish in ground cavities, equipment, or hollow trees.",
        "management": [
            "Do not disturb any bee colony — call a licensed beekeeper or pest control professional immediately",
            "Inspect farm equipment, irrigation boxes, and hollow trees regularly for new colonies",
            "Seal potential nesting sites (holes in walls, empty boxes, tyres) before colonies establish",
            "Wear full protective gear when working near known colonies",
        ],
        "chemical": [
            "Pyrethroid-based aerosol sprays for emergency colony knockdown — only by trained personnel",
            "Contact a licensed pest control operator for colony removal",
        ],
        "prevention": [
            "Install European honey bee colonies on the farm — they compete for nest sites and are far less aggressive",
            "Train all farm workers to recognise and respond to bee threats",
            "Keep emergency epinephrine (EpiPen) on site for allergic reactions",
        ],
    },
    "Aphids": {
        "display": "Aphids",
        "crops": "Almost all crops — especially brassicas, legumes, cereals, fruit trees",
        "severity": "Moderate",
        "description": "Tiny soft-bodied insects that cluster on new growth, sucking plant sap. Cause leaf curl, stunted growth, and honeydew which promotes sooty mould. Also vectors of over 100 plant viruses — making them more dangerous than their direct feeding damage suggests.",
        "management": [
            "Blast colonies off with a strong jet of water — effective and free",
            "Apply insecticidal soap (2% solution) directly to colonies every 5–7 days",
            "Neem oil spray (2%) disrupts aphid life cycle and acts as a repellent",
            "Introduce or encourage ladybirds, lacewings, and parasitic wasps — natural predators",
        ],
        "chemical": [
            "Imidacloprid systemic insecticide — absorbed by plant, kills aphids feeding on sap",
            "Pymetrozine — selective aphicide with low impact on beneficial insects",
            "Avoid broad-spectrum pyrethroids which kill natural predators and worsen aphid outbreaks long-term",
        ],
        "prevention": [
            "Use reflective silver mulch — disorients aphids and reduces landing rates by up to 50%",
            "Plant banker plants (e.g. barley with aphid colonies) to maintain predator populations between crops",
            "Avoid excess nitrogen fertiliser — lush, soft growth is most attractive to aphids",
            "Monitor weekly from crop establishment — early intervention prevents exponential population growth",
        ],
    },
    "Armyworms": {
        "display": "Armyworms",
        "crops": "Maize, wheat, sorghum, rice, pasture grasses",
        "severity": "High",
        "description": "Larvae of noctuid moths that feed in large groups, stripping leaves and stems. Named for their habit of moving en masse across fields when food runs out. A single generation can devastate an entire field within days.",
        "management": [
            "Scout fields at dawn and dusk when larvae are active — look for egg masses on leaves",
            "Hand-pick small infestations and destroy egg masses",
            "Apply Bacillus thuringiensis (Bt) spray — effective against young larvae, safe for beneficials",
            "Parasitic wasps (Cotesia species) are highly effective biological control agents",
        ],
        "chemical": [
            "Chlorpyrifos or lambda-cyhalothrin — apply in the evening when larvae are feeding",
            "Spinosad — derived from soil bacteria, effective and lower environmental impact",
            "Treat when larval density exceeds 2–3 per square metre in cereals",
        ],
        "prevention": [
            "Deep plough after harvest to expose pupae to predators and desiccation",
            "Install pheromone traps to monitor adult moth flights and predict outbreak timing",
            "Maintain field margins with native grasses to support parasitoid wasp populations",
            "Early planting reduces vulnerability — young armyworms prefer older, tougher crop growth",
        ],
    },
    "Brown Marmorated Stink Bugs": {
        "display": "Brown Marmorated Stink Bugs",
        "crops": "Apples, pears, peaches, sweet corn, soybeans, tomatoes, peppers",
        "severity": "High",
        "description": "Invasive shield bug (Halyomorpha halys) that pierces fruit and seeds, causing cat-facing, dimpling, and internal corking that renders produce unmarketable. Overwinters in large numbers in buildings and field shelters. Emits a strong odour when disturbed.",
        "management": [
            "Kaolin clay barrier spray on fruit — physical deterrent, wash off before harvest",
            "Pyrethrin spray on orchard perimeters as a knockdown measure",
            "Samurai wasp (Trissolcus japonicus) is an emerging biological control — check local availability",
            "Vacuum adults from trees in the morning when they are sluggish",
        ],
        "chemical": [
            "Bifenthrin or zeta-cypermethrin on orchard borders — focus on field edges where bugs enter",
            "Dinotefuran — effective systemic option for tree fruits",
            "Rotate chemical classes to prevent resistance development",
        ],
        "prevention": [
            "Seal all building entry points before October to prevent overwintering aggregations",
            "Use exclusion netting over high-value fruit crops during peak infestation periods (August–October)",
            "Monitor with commercially available pheromone traps from July onwards",
            "Remove wild hosts (tree of heaven, paulownia) near farm boundaries",
        ],
    },
    "Cabbage Loopers": {
        "display": "Cabbage Loopers",
        "crops": "Brassicas (cabbage, broccoli, cauliflower, kale), lettuce, celery, cotton",
        "severity": "Moderate",
        "description": "Larvae of Trichoplusia ni moths that loop their bodies as they walk. Feed voraciously on leaf tissue, creating large irregular holes. Greenish caterpillars that are well-camouflaged against foliage. Multiple generations per year in warm climates.",
        "management": [
            "Hand-pick larvae and egg masses from undersides of leaves",
            "Bacillus thuringiensis var. kurstaki (Btk) spray — highly effective on young larvae",
            "Introduce Trichogramma wasps to parasitise eggs",
            "Spinosad spray — effective and approved for organic production",
        ],
        "chemical": [
            "Indoxacarb — highly effective, low mammalian toxicity",
            "Methoxyfenozide — insect growth regulator, disrupts moulting",
            "Resistance to pyrethroids is widespread — avoid as primary treatment",
        ],
        "prevention": [
            "Row covers from transplanting exclude adult moths entirely",
            "Pheromone traps to monitor adult populations and time spray applications",
            "Interplant with dill, fennel, or yarrow to attract natural enemies",
            "Rotate brassica crops with non-host families each season",
        ],
    },
    "Citrus Canker": {
        "display": "Citrus Canker (bacterial disease)",
        "crops": "All citrus — oranges, lemons, limes, grapefruit, tangerines",
        "severity": "High",
        "description": "Bacterial disease (Xanthomonas citri) causing raised, corky lesions with yellow halos on leaves, stems, and fruit. Spreads rapidly via wind-driven rain, insects, and farm equipment. Reduces fruit quality and marketability; can cause premature fruit drop.",
        "management": [
            "Remove and destroy all infected plant material — bag and burn, do not compost",
            "Apply copper-based bactericide (copper hydroxide or copper oxychloride) after pruning and rain events",
            "Sterilise all pruning tools between trees with 70% isopropyl alcohol",
            "Establish windbreaks to reduce wind-driven rain spread between trees",
        ],
        "chemical": [
            "Copper hydroxide or copper oxychloride sprays every 3–4 weeks during wet season",
            "Streptomycin (where permitted) for severe outbreaks — check local regulations",
            "Apply before and after rain events for maximum protection",
        ],
        "prevention": [
            "Plant certified canker-free nursery stock only",
            "Quarantine new trees for 6 months before introducing to orchards",
            "Control Asian citrus psyllid — it creates wounds that allow bacterial entry",
            "Avoid working in wet orchards — bacteria spread readily on wet foliage",
        ],
    },
    "Colorado Potato Beetles": {
        "display": "Colorado Potato Beetles",
        "crops": "Potato, tomato, eggplant, pepper (solanaceous crops)",
        "severity": "High",
        "description": "Leptinotarsa decemlineata — one of the world's most damaging agricultural pests. Both adults and larvae defoliate plants rapidly. Notorious for developing resistance to virtually every class of insecticide — over 50 compounds to date.",
        "management": [
            "Hand-pick adults, larvae, and bright orange egg masses from leaf undersides daily",
            "Bacillus thuringiensis var. tenebrionis (Btt) — effective on young larvae only",
            "Spinosad spray — effective and rotation-worthy with other modes of action",
            "Release Edovum puttleri parasitoid wasps for egg mass control",
        ],
        "chemical": [
            "Imidacloprid seed treatment — systemic protection from emergence",
            "Cyantraniliprole — newer diamide insecticide with good efficacy",
            "Strict rotation of chemical classes is essential — resistance develops within 2–3 seasons",
        ],
        "prevention": [
            "Crop rotation is the single most effective tool — move potatoes at least 500m from previous year's field",
            "Straw mulch reduces adult beetle colonisation by 75% in some studies",
            "Plant early-maturing varieties to escape peak larval pressure",
            "Trench barriers around field perimeter trap walking adults before they reach crop",
        ],
    },
    "Corn Borers": {
        "display": "Corn Borers (European Corn Borer)",
        "crops": "Maize (corn), sorghum, cotton, peppers, potatoes",
        "severity": "High",
        "description": "Ostrinia nubilalis larvae bore into stalks, tassels, and ears of maize, causing lodging, ear rot, and significant yield loss. Entry wounds also allow fungal pathogens (including aflatoxin-producing Aspergillus) to infect the plant.",
        "management": [
            "Apply Bacillus thuringiensis (Bt) to whorls when egg masses are hatching",
            "Trichogramma ostriniae parasitoid wasps target egg masses effectively",
            "Time foliar sprays to coincide with egg hatch — once larvae bore inside, sprays are ineffective",
            "Spinosad or chlorantraniliprole applied to silks prevent ear damage",
        ],
        "chemical": [
            "Chlorantraniliprole (Coragen) — highly effective diamide insecticide, low bee toxicity",
            "Lambda-cyhalothrin applied at silk emergence",
            "Bt maize varieties provide season-long protection without spraying",
        ],
        "prevention": [
            "Shred and incorporate crop residue immediately after harvest — destroys overwintering pupae",
            "Plant Bt-traited maize varieties where available",
            "Monitor with pheromone traps to identify peak adult flight and optimise spray timing",
            "Late planting reduces first-generation pressure in some regions",
        ],
    },
    "Corn Earworms": {
        "display": "Corn Earworms (Tomato Fruitworms)",
        "crops": "Maize, tomato, cotton, soybean, sorghum",
        "severity": "High",
        "description": "Helicoverpa zea larvae feed on maize silk and kernels at the tip of the ear, and on tomato fruit. One of the most economically damaging insects in North American agriculture. High resistance to pyrethroids and some other insecticide classes.",
        "management": [
            "Apply mineral oil or Btk to maize silks every 2–3 days during silk emergence",
            "Trichogramma pretiosum parasitoid wasps for egg control",
            "Spinosad spray on tomato fruit when small larvae are first detected",
            "Monitor with pheromone traps to track adult flights",
        ],
        "chemical": [
            "Chlorantraniliprole — most effective current option, low resistance risk",
            "Spinetoram — effective on young larvae before they bore into ears",
            "Avoid pyrethroids — widespread resistance makes them largely ineffective",
        ],
        "prevention": [
            "Tight-husked maize varieties reduce ear penetration",
            "Pheromone trap network across farm for early warning",
            "Bt maize varieties provide excellent earworm control",
            "Destroy crop residue promptly after harvest to reduce pupation sites",
        ],
    },
    "Fall Armyworms": {
        "display": "Fall Armyworms",
        "crops": "Maize, sorghum, millet, rice, sugarcane, pasture",
        "severity": "Critical",
        "description": "Spodoptera frugiperda — a devastating migratory pest that has spread from the Americas to Africa and Asia since 2016, threatening food security for hundreds of millions of people. Larvae feed in the whorl of young maize plants, causing characteristic ragged leaf damage and frass.",
        "management": [
            "Scout maize fields from emergence — check whorl for frass and feeding damage",
            "Apply Bacillus thuringiensis (Bt) or Metarhizium anisopliae to whorls when infestation is detected",
            "Spinosad or emamectin benzoate — effective on young larvae in whorl stage",
            "Sand or fine ash poured into whorl physically dislodges and abrades young larvae",
        ],
        "chemical": [
            "Emamectin benzoate — highly effective, low resistance reported",
            "Chlorantraniliprole — excellent efficacy on young larvae",
            "Spinetoram or indoxacarb as alternatives in rotation",
            "Apply in the evening when larvae are active and before they bore deeper into the plant",
        ],
        "prevention": [
            "Early planting — crops that reach V6 stage early are more tolerant of damage",
            "Push-pull intercropping (maize + Desmodium + Napier grass border) reduces infestation by up to 86%",
            "Plant Bt maize varieties where available and affordable",
            "Conserve natural enemies: ground beetles, spiders, parasitic wasps are important suppressors",
        ],
    },
    "Fruit Flies": {
        "display": "Fruit Flies (Tephritidae)",
        "crops": "Mango, citrus, stone fruits, guava, tomato, capsicum",
        "severity": "High",
        "description": "Tephritid fruit flies (Bactrocera, Ceratitis species) lay eggs under fruit skin; larvae feed inside, causing premature fruit drop and complete internal destruction. A major quarantine pest that restricts market access internationally.",
        "management": [
            "Protein bait sprays (hydrolysed protein + malathion) — highly attractive to adult females",
            "Mass trapping with methyl eugenol or cuelure lures (species-specific attractants)",
            "Bag individual fruit clusters on high-value crops",
            "Collect and destroy all fallen fruit daily — do not leave on ground",
        ],
        "chemical": [
            "Malathion bait spray (GF-120 or equivalent) — spot sprays on foliage attract and kill adults",
            "Spinosad bait spray — approved for organic production",
            "Cover sprays of lambda-cyhalothrin at fruit set for high-pressure situations",
        ],
        "prevention": [
            "Sterile insect technique (SIT) in area-wide management programmes — highly effective",
            "Harvest fruit promptly at correct maturity — do not leave overripe fruit on tree",
            "Hot water treatment (46–47°C for 60 min) for export fruit disinfection",
            "Maintain farm hygiene — rotting fruit is the primary breeding site",
        ],
    },
    "Spider Mites": {
        "display": "Spider Mites (Two-spotted Spider Mite)",
        "crops": "Almost all crops — especially maize, beans, strawberries, tomatoes, cucumbers under stress",
        "severity": "Moderate",
        "description": "Tetranychus urticae — tiny eight-legged arachnids (not insects) that feed on leaf undersides, causing stippling, bronzing, and defoliation. Reproduce extremely rapidly in hot, dry conditions — populations can double in 3–5 days. Outbreaks are often triggered by broad-spectrum insecticide use that kills natural enemies.",
        "management": [
            "Spray leaf undersides forcefully with water — physically removes mites",
            "Predatory mites (Phytoseiulus persimilis, Neoseiulus californicus) — extremely effective biocontrol",
            "Neem oil or insecticidal soap sprays every 5–7 days targeting leaf undersides",
            "Maintain plant vigour with adequate irrigation — stressed plants are far more susceptible",
        ],
        "chemical": [
            "Abamectin — most effective miticide, use as a rescue treatment",
            "Bifenazate or hexythiazox — selective miticides with low impact on beneficials",
            "Avoid pyrethroids — they kill predatory mites and cause spider mite resurgence",
            "Rotate modes of action to prevent resistance (resistance develops very rapidly)",
        ],
        "prevention": [
            "Avoid dusty conditions — dust suppresses predatory mites and favours spider mites",
            "Maintain adequate irrigation especially in hot weather",
            "Establish banker plants with predatory mite colonies before crop is planted",
            "Minimise broad-spectrum insecticide use which destroys natural enemy populations",
        ],
    },
    "Thrips": {
        "display": "Thrips (Western Flower Thrips)",
        "crops": "Onions, peppers, cucumbers, strawberries, ornamentals, cotton",
        "severity": "Moderate",
        "description": "Frankliniella occidentalis — tiny slender insects (1–2mm) that rasp leaf and flower tissue, causing silvery scarring, distorted growth, and flower abortion. More damaging as vectors of Tomato Spotted Wilt Virus (TSWV) and Impatiens Necrotic Spot Virus than by direct feeding.",
        "management": [
            "Blue sticky traps — thrips are attracted to blue (unlike most pests which prefer yellow)",
            "Predatory mites (Amblyseius cucumeris) — specifically target thrips larvae",
            "Spinosad spray — very effective against thrips, approved for organic use",
            "Neem oil disrupts thrips development and acts as a repellent",
        ],
        "chemical": [
            "Spinosad or spinetoram — most effective option, rotate with other classes",
            "Cyantraniliprole — effective on pupating larvae in soil stage",
            "Abamectin — targets leaf-feeding adults",
            "Resistance to many insecticides is common — rotate chemical classes strictly",
        ],
        "prevention": [
            "Remove and destroy crop debris promptly — thrips pupate in soil under crop",
            "Use virus-resistant varieties where available (especially for TSWV-susceptible crops)",
            "Reflective silver mulch reduces thrips landing on plants by up to 40%",
            "Monitor with blue sticky traps from crop establishment to detect early",
        ],
    },
    "Tomato Hornworms": {
        "display": "Tomato Hornworms",
        "crops": "Tomato, pepper, eggplant, potato (solanaceous crops)",
        "severity": "Moderate",
        "description": "Manduca quinquemaculata — large green caterpillars (up to 10cm) with a distinctive horn at the tail end. Extremely well camouflaged against foliage. A single larva can defoliate a tomato plant within days. Related to tobacco hornworm.",
        "management": [
            "Hand-pick larvae — most effective method given their large size and individual damage",
            "Leave larvae with white cocoons attached — these are Braconid wasp pupae parasitising the hornworm",
            "Bacillus thuringiensis (Btk) spray — effective on young larvae before they reach full size",
            "Spinosad spray as an alternative to Bt",
        ],
        "chemical": [
            "Indoxacarb or chlorantraniliprole for severe infestations",
            "Avoid broad-spectrum insecticides which kill Braconid wasps — the most effective natural control",
        ],
        "prevention": [
            "Till soil after harvest to expose overwintering pupae to birds and frost",
            "Interplant with basil, borage, or marigolds which deter adult moths from laying eggs",
            "Pheromone traps for adult moth monitoring",
            "Encourage wasps, birds, and ground beetles — all important natural enemies",
        ],
    },
    "Western Corn Rootworms": {
        "display": "Western Corn Rootworms",
        "crops": "Maize (corn) — one of the most damaging maize pests in the world",
        "severity": "High",
        "description": "Diabrotica virgifera virgifera larvae feed on maize roots causing lodging (plants fall over — 'goosenecking'), reduced water and nutrient uptake, and severe yield loss. Adults feed on silk, tassels, and leaves. Notorious for developing resistance to both insecticides and Bt traits through crop rotation behaviour changes.",
        "management": [
            "Crop rotation is the primary management tool — rootworms cannot survive in non-corn crops",
            "Soil-applied insecticides at planting protect roots in first-year corn",
            "Biological control with entomopathogenic nematodes (Steinernema, Heterorhabditis)",
            "Adult population monitoring with yellow sticky traps during July–August",
        ],
        "chemical": [
            "Tefluthrin or chlorpyrifos soil insecticide applied in-furrow at planting",
            "Foliar pyrethroid sprays for adult silk feeding if threshold exceeded (1–2 beetles per plant)",
            "Bt traits (Cry3Bb1, mCry3A) in seed — rotate traits to manage resistance",
        ],
        "prevention": [
            "Annual crop rotation with soybean, wheat, or alfalfa — eliminates rootworm problem in rotated fields",
            "Avoid continuous maize production — rootworm populations explode without rotation",
            "Where rotation is not possible, use pyramided Bt traits combined with soil insecticide",
            "Monitor adult populations in August to predict larval pressure the following season",
        ],
    },
}

# Module-level model state — populated by load_pest_model() at startup.
_pest_model = None        # ResNet-18 nn.Module in eval mode
_pest_classes = None      # List[str] mapping output index → class name
_pest_model_meta = {"loaded": False, "val_acc": None, "n_classes": 0}


def load_pest_model() -> bool:
    """Load the pest ResNet-18 checkpoint into memory.

    Reads pest_model.pth and pest_class_names.json from MODELS_DIR. Populates
    module-level globals so run_pest_inference() can run without re-loading.

    Returns True on success, False if files are missing or the checkpoint is
    corrupt. torch is imported here (not at module level) to avoid a
    LightGBM+PyTorch segfault on macOS.
    """
    global _pest_model, _pest_classes, _pest_model_meta

    model_path = MODELS_DIR / "pest_model.pth"
    class_path = MODELS_DIR / "pest_class_names.json"

    print(f"Looking for pest model at: {model_path}")

    if not model_path.exists():
        print("   Not found — run: python train_pest_model.py --data '/Users/gargipande/Downloads/farm_insects'")
        return False
    if not class_path.exists():
        print("   pest_class_names.json not found")
        return False

    try:
        # Deferred imports — same macOS OpenMP segfault reason as disease_detection.
        import torch
        import torch.nn as nn
        from torchvision import models as tv_models

        # weights_only=False needed because the checkpoint contains custom metadata.
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        n_classes = checkpoint["n_classes"]

        # Load the ordered list of class names saved during training.
        with open(class_path) as f:
            _pest_classes = json.load(f)

        # Reconstruct the same architecture used in train_pest_model.py.
        net = tv_models.resnet18(weights=None)
        net.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(net.fc.in_features, n_classes),
        )
        net.load_state_dict(checkpoint["model_state"])
        net.eval()  # disable dropout and batch-norm training behaviour

        _pest_model = net
        _pest_model_meta = {
            "loaded": True,
            "val_acc": checkpoint.get("val_acc"),
            "n_classes": n_classes,
            "epoch": checkpoint.get("epoch"),
        }
        print(f"Pest model loaded — val_acc={checkpoint.get('val_acc', '?'):.4f}, {n_classes} classes")
        return True
    except Exception as e:
        print(f"Failed to load pest model: {e}")
        return False


def run_pest_inference(image_bytes: bytes) -> dict:
    """Run synchronous pest classification on image bytes.

    Lazily loads the model if not already loaded. Returns a dict with
    prediction, confidence, top-3 alternatives, management advice, and
    model metadata. Raises HTTP 503 if the model file is missing.

    Called inside run_in_executor so it does not block the async event loop.
    """
    import torch
    import torch.nn.functional as F

    # Lazy-load if startup loading was skipped or failed.
    if not _pest_model_meta["loaded"]:
        ok = load_pest_model()
        if not ok:
            raise HTTPException(
                status_code=503,
                detail="Pest model not trained yet. Run: python train_pest_model.py",
            )

    tensor = preprocess_image(image_bytes)  # shared ImageNet preprocessing from disease_detection

    # Run forward pass with gradient tracking disabled — saves memory and speeds up inference.
    with torch.no_grad():
        logits = _pest_model(tensor)
        probs = F.softmax(logits, dim=1)[0]  # convert raw scores to probabilities

    # Pull top-3 predictions; guard against models with fewer than 3 classes.
    top3_vals, top3_idx = torch.topk(probs, k=min(3, len(_pest_classes)))

    top_class = _pest_classes[top3_idx[0].item()]
    top_confidence = round(float(top3_vals[0]) * 100, 1)

    # Build the runner-up list (indices 1 and 2) shown in the UI as alternatives.
    alternatives = [
        {
            "class": _pest_classes[idx.item()],
            "display": PEST_INFO.get(_pest_classes[idx.item()], {}).get(
                "display", _pest_classes[idx.item()]
            ),
            "confidence": round(float(val) * 100, 1),
        }
        for val, idx in zip(top3_vals[1:], top3_idx[1:])
    ]

    # Look up pest info; fall back gracefully if the class has no entry in PEST_INFO.
    info = PEST_INFO.get(
        top_class,
        {
            "display": top_class,
            "crops": "Unknown",
            "severity": "Unknown",
            "description": "No specific information available for this pest.",
            "management": [],
            "chemical": [],
            "prevention": [],
        },
    )

    return {
        "predicted_class": top_class,
        "confidence": top_confidence,
        "display_name": info["display"],
        "crops": info["crops"],
        "severity": info["severity"],
        "severity_colour": SEVERITY_COLOUR.get(info["severity"], "#888"),  # hex for UI badge
        "description": info["description"],
        "management": info["management"],
        "chemical": info["chemical"],
        "prevention": info["prevention"],
        "alternatives": alternatives,
        "model_accuracy": _pest_model_meta,
    }


async def identify_pest(file: UploadFile = File(...)):
    """FastAPI endpoint: identify a farm pest from an uploaded image.

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
        result = await loop.run_in_executor(None, run_pest_inference, image_bytes)
    except HTTPException:
        raise  # pass through 503 (model not found) unchanged
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pest identification error: {str(e)}")

    return result


def pest_model_status():
    """Return the current model metadata dict (loaded flag, val_acc, n_classes)."""
    return _pest_model_meta
