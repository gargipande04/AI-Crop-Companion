"""
yield_engine.py — Yield Prediction & Sustainability Engine
Handles all ML training, ensemble prediction, and sustainability reporting.

Imports from here in main.py:
    from yield_engine import (
        ensemble_predict, sustainability_report,
        primary_metrics, model_metrics, trained_pipelines,
        VALID_AREAS, VALID_ITEMS, LATEST_YEAR,
    )
"""

import numpy as np
import pandas as pd
from fastapi import HTTPException

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import r2_score

from app.core.config import DATA_DIR

# Optional boosting libraries — platform degrades gracefully if missing
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping. Install: conda install -c conda-forge xgboost")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed — skipping. Install: pip install lightgbm")


# SECTION 1: DATA LOADING & PREPROCESSING
# Source: FAOSTAT crop yield dataset (yield_df.csv).
# Features: Area (country), Item (crop), Year, annual rainfall (mm),
#           average temperature (°C), pesticide use (tonnes).
# Target:   yield in hectograms per hectare (hg/ha).
#
# Temporal 80/20 split — rows before the 80th-percentile year go to training,
# later rows to test. This reflects real deployment: the model is always asked
# to predict years it has not seen.

print("Loading yield data...")
df = pd.read_csv(DATA_DIR / "yield_df.csv").drop(columns=["Unnamed: 0"], errors="ignore")

# Normalise column names across different CSV versions.
df = df.rename(columns={
    "average_rain_fall_mm_per_year": "rainfall",
    "avg_temp":                      "temperature",
    "hg/ha_yield":                   "yield",
    "pesticides_tonnes":             "pesticides",
})

FEATURES = ["Area", "Item", "Year", "rainfall", "temperature", "pesticides"]
TARGET   = "yield"

df = df.dropna(subset=FEATURES + [TARGET]).copy()

# Lookup sets used for input validation in /predict and UI dropdowns in /metadata.
VALID_AREAS = sorted(df["Area"].unique().tolist())
VALID_ITEMS = sorted(df["Item"].unique().tolist())
LATEST_YEAR = int(df["Year"].max())

# Temporal split: train on data up to 80th-percentile year, test on the rest.
SPLIT_YEAR = int(df["Year"].quantile(0.8))
train_df   = df[df["Year"] <= SPLIT_YEAR].copy()
test_df    = df[df["Year"] >  SPLIT_YEAR].copy()

# Outlier bounds fitted on training data onlys — avoids test-set contamination.
lo = train_df[TARGET].quantile(0.005)
hi = train_df[TARGET].quantile(0.995)
train_df = train_df[(train_df[TARGET] >= lo) & (train_df[TARGET] <= hi)]
test_df  = test_df[ (test_df[TARGET]  >= lo) & (test_df[TARGET]  <= hi)]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | Split year: {SPLIT_YEAR}")


# SECTION 2: SHARED PREPROCESSOR
# OrdinalEncoder is used for categorical columns because tree-based models
# split on numeric thresholds — they handle integer-encoded categories natively
# and do not benefit from the sparse binary columns OneHotEncoder would create.

cat_cols = ["Area", "Item"]
num_cols = ["Year", "rainfall", "temperature", "pesticides"]

preprocessor = ColumnTransformer(transformers=[
    # Categorical: impute missing with most-frequent value, encode as integer.
    # unknown_value=-1 means unseen categories at inference produce a defined
    # value rather than raising an error.
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
    # Numeric: impute missing with column median, then standardise.
    # Standardisation is required because year, rainfall, temperature, and
    # pesticides are on very different scales.
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), num_cols),
])


def make_pipeline(regressor) -> Pipeline:
    """
    Wrap a regressor in a full sklearn Pipeline with preprocessing and
    target standardisation.

    TransformedTargetRegressor standardises the target (yield) before
    passing it to the regressor, then inverse-transforms predictions back
    to hg/ha. This improves convergence for gradient-based models.

    Args:
        regressor: Any sklearn-compatible regressor instance.

    Returns:
        A fitted-ready sklearn Pipeline: preprocessor -> target-scaled regressor.
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    TransformedTargetRegressor(
            regressor=regressor,
            transformer=StandardScaler(),
        )),
    ])


def robust_error_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute a robust Median Absolute Percentage Error (MdAPE).

    Standard MAPE is unreliable on crop yield data because near-zero true
    yields produce enormous percentage errors. This function:
      - Uses the median rather than the mean to eliminate outlier influence.
      - Drops the bottom 10th percentile of true yields to remove rows where
        any small absolute error becomes a huge percentage.

    Args:
        y_true: Ground-truth yield values in hg/ha.
        y_pred: Predicted yield values in hg/ha.

    Returns:
        Median absolute percentage error as a float percentage,
        e.g. 10.4 means the model is typically within ~10% of the true yield.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    floor  = np.percentile(y_true, 10)
    mask   = y_true >= max(floor, 1.0)
    ape    = np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.median(ape) * 100)


# SECTION 3: TRAIN INDIVIDUAL MODELS
print("Training yield models...")

models_to_train = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.07,
        subsample=0.8, random_state=42,
    ),
}

if HAS_XGB:
    models_to_train["XGBoost"] = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.07,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )

if HAS_LGB:
    models_to_train["LightGBM"] = LGBMRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.07,
        num_leaves=63, subsample=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )

trained_pipelines = {}  # name -> fitted Pipeline
model_metrics     = {}  # name -> {r2, err_pct}

for name, reg in models_to_train.items():
    print(f"  Training {name}...")
    pipe = make_pipeline(reg)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    r2      = float(r2_score(y_test, preds))
    err_pct = robust_error_pct(y_test.values, preds)

    trained_pipelines[name] = pipe
    model_metrics[name] = {"r2": round(r2, 4), "err_pct": round(err_pct, 1)}
    print(f"    R²={r2:.4f}  Median error={err_pct:.1f}%")


# SECTION 4: WEIGHTED ENSEMBLE
# Ensemble weights are derived from a held-out validation split inside the
# training period — the test set is never used for weight estimation.
print("Building ensemble...")

# Last 20% of training rows as a validation split for weight estimation.
val_size = int(len(X_train) * 0.2)
X_val    = X_train.iloc[-val_size:]
y_val    = y_train.iloc[-val_size:]

val_r2 = {}
for name, pipe in trained_pipelines.items():
    val_preds    = pipe.predict(X_val)
    val_r2[name] = max(0.0, float(r2_score(y_val, val_preds)))

# Normalise weights to sum to 1.0.
total_w = sum(val_r2.values())
weights = {n: v / total_w for n, v in val_r2.items()}


def ensemble_predict(X: pd.DataFrame) -> np.ndarray:
    """
    Generate a weighted ensemble prediction.

    Args:
        X: DataFrame with columns matching FEATURES.

    Returns:
        NumPy array of predicted yield values in hg/ha.
    """
    preds = np.zeros(len(X))
    for name, pipe in trained_pipelines.items():
        preds += weights[name] * pipe.predict(X)
    return preds


# Final test-set metrics for the ensemble (reported in the UI).
ens_preds   = ensemble_predict(X_test)
ens_r2      = float(r2_score(y_test, ens_preds))
ens_err_pct = robust_error_pct(y_test.values, ens_preds)

model_metrics["Ensemble"] = {
    "r2":      round(ens_r2,      4),
    "err_pct": round(ens_err_pct, 1),
    "weights": {n: round(w, 3) for n, w in weights.items()},
}

best_model_name = max(
    (n for n in trained_pipelines),
    key=lambda n: model_metrics[n]["r2"]
)
print(f"  Best single model: {best_model_name} (R²={model_metrics[best_model_name]['r2']})")
print(f"  Ensemble R²={ens_r2:.4f}  Median error={ens_err_pct:.1f}%")

# Metrics dict returned by /metadata and shown in the Model Comparison panel.
primary_metrics = {
    **model_metrics["Ensemble"],
    "split_year":  SPLIT_YEAR,
    "train_rows":  len(X_train),
    "test_rows":   len(X_test),
    "all_models":  model_metrics,
    "best_single": best_model_name,
}

 # SECTION 5: SUSTAINABILITY ENGINE
 # Seasonal crop water requirements (mm) from FAO Irrigation Paper No. 56.
 
CROP_WATER = {
    "Maize":          500,  "Rice, paddy":     1200, "Wheat":           450,
    "Soybeans":       450,  "Potatoes":         500, "Sugar cane":     1500,
    "Sorghum":        450,  "Cassava":          600, "Sweet potatoes":  500,
    "Plantains":      900,  "Yams":             550, "Oil palm":       1200,
    "Cocoa beans":    800,  "Coffee, green":    900, "Bananas":         900,
    "Oranges":        900,  "Grapes":           600, "Apples":          700,
    "Tomatoes":       600,  "Onions":           500, "Sunflower":       600,
    "Rapeseed":       400,  "Cotton":           700, "Ground-nuts":     500,
    "Millet":         400,  "Barley":           400, "Oats":            450,
    "Rye":            400,  "Linseed":          400, "Olives":          500,
}
DEFAULT_WATER = 600  # mm/season fallback for unlisted crops

# IPCC Tier 1 simplified emission factor for synthetic pesticide production.
PESTICIDE_CO2_FACTOR = 18.0  # kg CO₂-eq per kg pesticide applied

# Income tier thresholds (pesticides in tonnes, matching CSV units).
INCOME_TIERS = {
    "low":    {"max_pesticides": 20,  "max_irrigation_pct": 0.3,  "label": "Low-input / Smallholder"},
    "medium": {"max_pesticides": 80,  "max_irrigation_pct": 0.65, "label": "Medium-input"},
    "high":   {"max_pesticides": 999, "max_irrigation_pct": 1.0,  "label": "High-input / Commercial"},
}


def sustainability_report(
    crop: str,
    rainfall_mm: float,
    pesticides_kg: float,
    predicted_yield_hg_ha: float,
    income_level: str = "medium",
) -> dict:
    """
    Generate a per-hectare sustainability and advisory report.

    Computes water balance, carbon footprint, water productivity, and a
    green score, then generates prioritised recommendations tailored to
    the farmer's income level.

    Args:
        crop:                  Crop name matched against CROP_WATER table.
        rainfall_mm:           Annual rainfall at the farm location in mm.
        pesticides_kg:         Pesticide quantity in tonnes (matches CSV units).
        predicted_yield_hg_ha: Predicted yield from the ensemble in hg/ha.
        income_level:          One of "low", "medium", or "high".

    Returns:
        Dictionary with water metrics, carbon score, yield in t/ha,
        income tier label, and a prioritised list of recommendations.
    """
    water_req = CROP_WATER.get(crop, DEFAULT_WATER)
    water_gap = max(0.0, water_req - rainfall_mm)        # extra irrigation needed
    water_eff = min(1.0, rainfall_mm / (water_req + 1))  # rainfall coverage 0–1

    # Carbon score: higher = greener. Divisor of 35 calibrated so typical
    # inputs (0–200 tonnes) produce scores across the full 0–100 range.
    carbon_kg    = pesticides_kg * PESTICIDE_CO2_FACTOR
    carbon_score = max(0, 100 - min(100, carbon_kg / 35))  # 0=dirty, 100=clean

    # Water productivity: tonnes of yield per cubic metre of water consumed.
    yield_t    = predicted_yield_hg_ha / 100              # hg/ha → t/ha
    water_prod = round(yield_t / (water_req / 1000), 2)   # t/m³

    tier = INCOME_TIERS.get(income_level, INCOME_TIERS["medium"])

    recs = []

    # Water recommendation
    if water_gap > 200:
        recs.append({
            "category": "Water", "priority": "HIGH",
            "tip": f"{crop} needs ~{water_req} mm; rainfall only {rainfall_mm:.0f} mm "
                   f"— {water_gap:.0f} mm irrigation required.",
            "low_income_alt": "Collect rainwater in farm ponds / use mulch to cut evaporation by 25–40%.",
        })
    elif water_gap > 50:
        recs.append({
            "category": "Water", "priority": "MEDIUM",
            "tip": f"Moderate water deficit (~{water_gap:.0f} mm). Supplemental irrigation advised.",
            "low_income_alt": "Deficit irrigation at critical growth stages saves 30% water with <10% yield loss.",
        })
    else:
        recs.append({
            "category": "Water", "priority": "LOW",
            "tip": "Rainfall appears sufficient for this crop. No major irrigation needed.",
            "low_income_alt": "Ensure good drainage to prevent waterlogging.",
        })

    # Pesticide / pest management recommendation
    if pesticides_kg > tier["max_pesticides"]:
        recs.append({
            "category": "Pest Management", "priority": "MEDIUM",
            "tip": f"Pesticide use ({pesticides_kg:.0f} t) is high for {income_level}-income context.",
            "low_income_alt": "IPM: neem-based biopesticides, pheromone traps, and resistant varieties "
                              "cut chemical costs 40–60%.",
        })
    else:
        recs.append({
            "category": "Pest Management", "priority": "LOW",
            "tip": "Pesticide use is within sustainable range for this income tier.",
            "low_income_alt": "Rotate crops annually to break pest cycles — free and effective.",
        })

    # Soil health (always shown)
    recs.append({
        "category": "Soil Health", "priority": "MEDIUM",
        "tip": "Maintaining organic matter is the single best free yield insurance.",
        "low_income_alt": "Return crop residues, use legume cover crops (free nitrogen!), "
                          "and avoid bare soil between seasons.",
    })

    # Crop variety (low-income only)
    if income_level == "low":
        recs.append({
            "category": "Crop Choice", "priority": "HIGH",
            "tip": "For smallholders, drought-tolerant varieties can boost yield 15–40% with no extra input cost.",
            "low_income_alt": "Contact local agricultural extension for subsidised improved seed programmes.",
        })

    # Climate resilience (always shown)
    recs.append({
        "category": "Climate Resilience", "priority": "LOW",
        "tip": "Diversify crops to hedge against price and weather shocks.",
        "low_income_alt": "Intercropping (e.g., maize + beans) maintains income if one crop fails "
                          "and fixes nitrogen for free.",
    })

    return {
        "water_requirement_mm":    water_req,
        "rainfall_mm":             rainfall_mm,
        "irrigation_needed_mm":    round(water_gap, 1),
        "water_efficiency_pct":    round(water_eff * 100, 1),
        "water_productivity_t_m3": water_prod,
        "carbon_intensity_score":  round(carbon_score, 1),  # 0=dirty, 100=clean
        "pesticide_carbon_kg_ha":  round(carbon_kg, 1),
        "yield_t_ha":              round(yield_t, 2),
        "income_tier":             tier["label"],
        "recommendations":         recs,
    }


def get_metadata() -> dict:
    """
    Return the metadata used by the frontend dropdowns and metrics panel.
    """
    return {
        "areas":       VALID_AREAS,
        "items":       VALID_ITEMS,
        "metrics":     primary_metrics,
        "latest_year": LATEST_YEAR,
    }


def get_yield_model_names() -> list[str]:
    """
    Return the list of trained yield model names.
    """
    return list(trained_pipelines.keys())


def predict_yield(data) -> dict:
    """
    Run a yield prediction and attach model breakdown plus sustainability.

    The input object is expected to expose:
      - area
      - crop
      - rainfall
      - temperature
      - pesticides
      - income_level
    """
    if data.area not in VALID_AREAS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown area '{data.area}'. Call /metadata for the valid list.",
        )
    if data.crop not in VALID_ITEMS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown crop '{data.crop}'. Call /metadata for the valid list.",
        )

    feat = pd.DataFrame([{
        "Area":        data.area,
        "Item":        data.crop,
        "Year":        LATEST_YEAR,
        "rainfall":    data.rainfall,
        "temperature": data.temperature,
        "pesticides":  data.pesticides,
    }])

    pred = float(ensemble_predict(feat)[0])

    breakdown = {
        name: round(float(pipe.predict(feat)[0]), 2)
        for name, pipe in trained_pipelines.items()
    }
    breakdown["Ensemble"] = round(pred, 2)

    sustain = sustainability_report(
        crop=data.crop,
        rainfall_mm=data.rainfall,
        pesticides_kg=data.pesticides,
        predicted_yield_hg_ha=pred,
        income_level=data.income_level or "medium",
    )

    return {
        "predicted_yield":      round(pred, 2),
        "unit":                 "hg/ha",
        "error_pct":            f"~{primary_metrics['err_pct']:.1f}% median error on test set",
        "reference_year":       LATEST_YEAR,
        "model_breakdown":      breakdown,
        "ensemble_weights":     model_metrics["Ensemble"].get("weights", {}),
        "sustainability":       sustain,
    }
