# AI Crop Companion

Crop Companion is a FastAPI application that combines:

- yield prediction from tabular crop/climate/input data
- sustainability guidance layered on top of the yield result
- plant disease detection from images using a ResNet-18 model
- pest identification from images using a ResNet-18 model
- an AI chatbot for agricultural Q&A through Anthropic

## Run

Install dependencies and start the app the same way as before:

```bash
python main.py
```

Then open:

- `http://127.0.0.1:8000/` for `home.html`
- `http://127.0.0.1:8000/app` for `index.html`

## Train The Disease And Pest Models

The image models are trained separately from the FastAPI app and save into [models](/Users/gargipande/AI%20crop%20guide/models) by default.

### Disease Model

Train the disease detector with:

```bash
python train_disease_model.py --data ./Diseases_Dataset --epochs 15
```

Expected dataset layout:

```text
Diseases_Dataset/
  Black_knot/
  Chlorosis/
  Dog_vomit_slime_mold/
  Elderberry_rust/
  Golden_canker/
  Gymnosporangium_Rusts/
  peach_leaf_curl/
  Powdery_Mildew/
  Sooty_Mold/
  Tar_Spot/
```

This creates:

- [models/disease_model.pth](/Users/gargipande/AI%20crop%20guide/models/disease_model.pth)
- [models/class_names.json](/Users/gargipande/AI%20crop%20guide/models/class_names.json)

### Pest Model

Train the pest detector with:

```bash
python train_pest_model.py --data ./farm_insects --epochs 20
```

This creates:

- [models/pest_model.pth](/Users/gargipande/AI%20crop%20guide/models/pest_model.pth)
- [models/pest_class_names.json](/Users/gargipande/AI%20crop%20guide/models/pest_class_names.json)

### Notes

- Both training scripts use transfer learning with `ResNet-18`.
- If you want a different output location, pass `--out <folder>`.
- The main app can start without these files, but `/diagnose` and `/identify-pest` will not work until the relevant model files exist.

## Project Structure

- [main.py](/Users/gargipande/AI%20crop%20guide/main.py): root entrypoint, used for `python main.py`
- [app/server.py](/Users/gargipande/AI%20crop%20guide/app/server.py): FastAPI app wiring and route registration
- [app/core/config.py](/Users/gargipande/AI%20crop%20guide/app/core/config.py): shared path and environment configuration
- [app/services/yield_engine.py](/Users/gargipande/AI%20crop%20guide/app/services/yield_engine.py): yield prediction and sustainability engine
- [app/services/disease_detection.py](/Users/gargipande/AI%20crop%20guide/app/services/disease_detection.py): disease detection service
- [app/services/pest_detection.py](/Users/gargipande/AI%20crop%20guide/app/services/pest_detection.py): pest detection service
- [app/services/ai_chatbot.py](/Users/gargipande/AI%20crop%20guide/app/services/ai_chatbot.py): AI chatbot service
- [templates/home.html](/Users/gargipande/AI%20crop%20guide/templates/home.html): landing page
- [templates/index.html](/Users/gargipande/AI%20crop%20guide/templates/index.html): application UI
- [assets/images](/Users/gargipande/AI%20crop%20guide/assets/images): image assets used by the frontend
- [assets/js/frontend.js](/Users/gargipande/AI%20crop%20guide/assets/js/frontend.js): frontend JavaScript asset
- [data](/Users/gargipande/AI%20crop%20guide/data): CSV datasets
- [models](/Users/gargipande/AI%20crop%20guide/models): trained `.pth` model files and class-name JSON files

## Notes

- The app launches with `python main.py`.
- Static assets are served from `/static/...`, with images under `/static/images/...`.
- Disease and pest inference expect the trained model files to be present in [models](/Users/gargipande/AI%20crop%20guide/models).
- The chat endpoint requires `ANTHROPIC_API_KEY` in the environment.


