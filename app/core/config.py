"""
config.py — Central path configuration for Crop Companion.

This module is imported first by all service modules. It sets the OpenMP
environment variables before any ML library (sklearn, XGBoost, LightGBM,
torch) is imported anywhere else in the app — that ordering is required to
prevent an OpenMP segfault on macOS when multiple libraries each try to
load their own OpenMP runtime.
"""

import os
from pathlib import Path

# Must be set before any sklearn / xgboost / lightgbm / torch import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Resolved absolute paths — safe to import anywhere without __file__ tricks.
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root
APP_DIR = Path(__file__).resolve().parents[1]        # app/
ASSETS_DIR = PROJECT_ROOT / "assets"                 # CSS, JS, images
IMAGES_DIR = ASSETS_DIR / "images"
JS_DIR = ASSETS_DIR / "js"
DATA_DIR = PROJECT_ROOT / "data"                     # yield_df.csv etc.
MODELS_DIR = PROJECT_ROOT / "models"                 # *.pth + class_names.json
TEMPLATES_DIR = PROJECT_ROOT / "templates"           # index.html, home.html
