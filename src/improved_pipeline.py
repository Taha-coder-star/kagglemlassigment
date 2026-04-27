"""Irrigation-need classifier pipeline (v2).

Changes from v1
---------------
- LightGBM backend (--model lgbm) with categorical support + early stopping
- Blend mode (--model blend): averages HistGBM + LightGBM probabilities
- 9 additional domain features (Penman-Monteith ETC proxy, crop Kc, soil WHC,
  drought index, water-stress index, net water balance …) → ~40 total
- Finer probability-multiplier grid (22 × 37 = 814 points vs. 325 in v1)
- Optional scipy.optimize for multiplier fine-tuning (--use-scipy-threshold)
- OOF calibration fix: multipliers tuned on a held-out calibration split
  (--oof-calib-size 0.15, default), not on the full OOF probability pool.
  Use --oof-tune-on-oof to restore the v1 behaviour.

Recommended Colab experiments (run one at a time, compare Kaggle scores)
-------------------------------------------------------------------------
# 1. Baseline reproduction
python src/improved_pipeline.py --holdout-only --model histgbm

# 2. LightGBM holdout  ← most likely improvement
python src/improved_pipeline.py --holdout-only --model lgbm

# 3. Blend holdout  ← ensemble diversity boost
python src/improved_pipeline.py --holdout-only --model blend

# 4. LightGBM OOF with calibration fix
python src/improved_pipeline.py --model lgbm

# 5. Blend OOF with calibration fix
python src/improved_pipeline.py --model blend

# 6. Scipy threshold fine-tuning (add to any of the above)
python src/improved_pipeline.py --holdout-only --model lgbm --use-scipy-threshold

# 7. Tweak High class weight (try 1.5, 2.0)
python src/improved_pipeline.py --holdout-only --model lgbm --high-weight-multiplier 1.5
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    warnings.warn("lightgbm not installed — --model lgbm/blend will fall back to HistGBM.")

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


SEED = 42
TARGET = "Irrigation_Need"
CLASS_NAMES = ["Low", "Medium", "High"]
LABEL_MAP = {label: idx for idx, label in enumerate(CLASS_NAMES)}
INV_MAP = {idx: label for label, idx in LABEL_MAP.items()}

# Crop evapotranspiration coefficients (FAO-56, simplified)
_CROP_KC = {
    "Rice": 1.20, "Sugarcane": 1.10, "Maize": 0.85,
    "Cotton": 0.90, "Wheat": 0.75, "Potato": 0.80,
}
# Growth-stage water demand multipliers
_GROWTH_KS = {
    "Vegetative": 0.85, "Flowering": 1.10,
    "Harvest": 0.60,   "Sowing": 0.70,
}
# Soil water-holding capacity scores
_SOIL_WHC = {
    "Clay": 1.20, "Loamy": 1.00, "Silt": 0.90, "Sandy": 0.65,
}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Irrigation-need classifier v2.")

    p.add_argument("--data-dir",   default="data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--valid-size", type=float, default=0.20,
                   help="Holdout validation fraction.")
    p.add_argument("--folds", type=int, default=5,
                   help="Stratified KFold splits for OOF mode.")

    # Model backend
    p.add_argument("--model", choices=["histgbm", "lgbm", "blend"],
                   default="histgbm",
                   help="'blend' averages HistGBM + LightGBM probabilities.")

    # HistGBM hyperparams
    p.add_argument("--max-iter",       type=int,   default=600)
    p.add_argument("--learning-rate",  type=float, default=0.05)
    p.add_argument("--max-leaf-nodes", type=int,   default=63)
    p.add_argument("--l2-reg",         type=float, default=0.02)
    p.add_argument("--min-samples-leaf", type=int, default=35)

    # LightGBM hyperparams
    p.add_argument("--lgb-n-estimators", type=int,   default=2000)
    p.add_argument("--lgb-lr",           type=float, default=0.04)
    p.add_argument("--lgb-num-leaves",   type=int,   default=127)
    p.add_argument("--lgb-subsample",    type=float, default=0.80)
    p.add_argument("--lgb-colsample",    type=float, default=0.70)
    p.add_argument("--lgb-reg-alpha",    type=float, default=0.05)
    p.add_argument("--lgb-reg-lambda",   type=float, default=0.20)
    p.add_argument("--lgb-min-child-samples", type=int, default=50)

    # Class-weight / threshold
    p.add_argument("--high-weight-multiplier", type=float, default=1.0,
                   help="Extra multiplier on balanced class weight for High.")
    p.add_argument("--use-scipy-threshold", action="store_true",
                   help="Fine-tune multipliers with Nelder-Mead after grid search.")

    # OOF calibration
    p.add_argument("--oof-calib-size", type=float, default=0.15,
                   help="Fraction of train held out for multiplier calibration in OOF mode.")
    p.add_argument("--oof-tune-on-oof", action="store_true",
                   help="Tune multipliers on the full OOF pool (v1 behaviour, can overfit).")

    # Misc
    p.add_argument("--sample-rows",    type=int, default=None,
                   help="Limit training rows for quick smoke tests.")
    p.add_argument("--skip-submission", action="store_true")
    p.add_argument("--holdout-only",    action="store_true",
                   help="Run single holdout split instead of OOF ensemble.")

    return p.parse_args()


# ─── Feature engineering ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    area     = df["Field_Area_hectare"].clip(lower=0.001)
    humidity = df["Humidity"].clip(lower=0.001)
    rainfall = df["Rainfall_mm"]
    moisture = df["Soil_Moisture"]
    temp     = df["Temperature_C"]
    prev_irr = df["Previous_Irrigation_mm"]

    # ── v1 features (unchanged) ──────────────────────────────────────────────
    df["Rainfall_per_hectare"]        = rainfall / area
    df["Total_Water_mm"]              = rainfall + prev_irr
    df["Rain_minus_Previous_Irrigation"] = rainfall - prev_irr
    df["Moisture_Deficit"]            = (50.0 - moisture).clip(lower=0.0)
    df["Heat_Dryness_Index"]          = temp * (100.0 - df["Humidity"]) / 100.0
    df["Evaporation_Proxy"]           = temp * df["Sunlight_Hours"] * df["Wind_Speed_kmh"] / humidity
    df["Soil_Water_Retention"]        = moisture * df["Organic_Carbon"]
    df["Salinity_Stress"]             = df["Electrical_Conductivity"] * temp
    df["Moisture_Rainfall_Ratio"]     = moisture / (rainfall + 1.0)

    df["Crop_Season"]   = df["Crop_Type"].astype(str) + "_" + df["Season"].astype(str)
    df["Soil_Irrigation"] = df["Soil_Type"].astype(str) + "_" + df["Irrigation_Type"].astype(str)
    df["Region_Water"]  = df["Region"].astype(str) + "_" + df["Water_Source"].astype(str)

    # ── v2 features ──────────────────────────────────────────────────────────

    # Simplified Penman–Monteith PET proxy (temperature + vapour deficit)
    sat_vp  = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))   # kPa
    delta   = 4098 * sat_vp / (temp + 237.3) ** 2              # kPa/°C
    df["PET_Proxy"] = (
        0.408 * delta * df["Sunlight_Hours"]
        + 0.665e-3 * df["Wind_Speed_kmh"] * sat_vp * (1 - df["Humidity"] / 100.0)
    ).clip(lower=0.0)

    # Crop and growth stage water demand coefficients
    df["Crop_Kc"]    = df["Crop_Type"].map(_CROP_KC).fillna(1.0)
    df["Growth_Ks"]  = df["Crop_Growth_Stage"].map(_GROWTH_KS).fillna(0.85)
    df["Soil_WHC"]   = df["Soil_Type"].map(_SOIL_WHC).fillna(1.0)

    # Adjusted crop evapotranspiration
    df["ETc"] = df["PET_Proxy"] * df["Crop_Kc"] * df["Growth_Ks"]

    # Net water balance (negative = irrigation deficit)
    df["Net_Water_Balance"] = rainfall + prev_irr - df["ETc"]

    # Water stress ratio (how well current supply covers demand)
    df["Water_Stress_Index"] = (rainfall + prev_irr + moisture) / (df["ETc"] + 1.0)

    # Drought index (temperature × sunshine / rainfall)
    df["Drought_Index"] = temp * df["Sunlight_Hours"] / (rainfall + 1.0)

    # Soil moisture adjusted for holding capacity
    df["Adj_Soil_Moisture"] = moisture * df["Soil_WHC"]

    # Per-hectare irrigation application
    df["Irrigation_per_ha"] = prev_irr / area

    # Two new categorical interactions
    df["Crop_IrrType"]   = df["Crop_Type"].astype(str) + "_" + df["Irrigation_Type"].astype(str)
    df["Soil_WaterSrc"]  = df["Soil_Type"].astype(str) + "_" + df["Water_Source"].astype(str)

    return df


def align_categorical_dtypes(
    train: pd.DataFrame, test: pd.DataFrame, categorical_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train.copy(), test.copy()
    for col in categorical_cols:
        vals = pd.concat(
            [train[col].astype(str), test[col].astype(str)], ignore_index=True
        )
        cats = pd.Index(vals.unique())
        train[col] = pd.Categorical(train[col].astype(str), categories=cats)
        test[col]  = pd.Categorical(test[col].astype(str),  categories=cats)
    return train, test


def make_feature_matrices(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    train_fe = add_features(train)
    test_fe  = add_features(test)

    cat_cols = [
        c for c in train_fe.select_dtypes(include=["object", "category"]).columns
        if c != TARGET
    ]
    train_fe, test_fe = align_categorical_dtypes(train_fe, test_fe, cat_cols)

    feature_cols = [c for c in train_fe.columns if c not in ("id", TARGET)]
    X      = train_fe[feature_cols]
    X_test = test_fe[feature_cols]
    y      = train_fe[TARGET].map(LABEL_MAP).to_numpy()

    return X, X_test, y, feature_cols


def make_class_weight(y: np.ndarray, high_multiplier: float) -> dict[int, float]:
    raw = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y)
    cw  = {idx: float(w) for idx, w in enumerate(raw)}
    cw[LABEL_MAP["High"]] *= high_multiplier
    return cw


# ─── Model builders ───────────────────────────────────────────────────────────

def build_histgbm(
    args: argparse.Namespace, class_weight: dict[int, float]
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_reg,
        max_bins=255,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=30,
        random_state=SEED,
        class_weight=class_weight,
    )


def build_lgbm(
    args: argparse.Namespace, class_weight: dict[int, float]
) -> "lgb.LGBMClassifier":
    # LightGBM expects string-keyed class_weight when labels are strings
    cw_named = {CLASS_NAMES[k]: v for k, v in class_weight.items()}
    return lgb.LGBMClassifier(
        n_estimators=args.lgb_n_estimators,
        learning_rate=args.lgb_lr,
        num_leaves=args.lgb_num_leaves,
        max_depth=-1,
        min_child_samples=args.lgb_min_child_samples,
        subsample=args.lgb_subsample,
        subsample_freq=1,
        colsample_bytree=args.lgb_colsample,
        reg_alpha=args.lgb_reg_alpha,
        reg_lambda=args.lgb_reg_lambda,
        class_weight=cw_named,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )


def _encode_cats_for_lgbm(X: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas Categorical columns to int codes for LightGBM."""
    X = X.copy()
    for col in X.select_dtypes(include="category").columns:
        X[col] = X[col].cat.codes
    return X


def _cat_feature_indices(X: pd.DataFrame) -> list[int]:
    return [i for i, col in enumerate(X.columns)
            if pd.api.types.is_categorical_dtype(X[col])]


# ─── Fit / predict dispatch ───────────────────────────────────────────────────

def fit_model(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
) -> None:
    if _HAS_LGB and isinstance(model, lgb.LGBMClassifier):
        X_tr_enc = _encode_cats_for_lgbm(X_train)
        cat_idx  = _cat_feature_indices(X_train)

        if X_val is not None and y_val is not None:
            X_va_enc = _encode_cats_for_lgbm(X_val)
            model.fit(
                X_tr_enc, [CLASS_NAMES[i] for i in y_train],
                eval_set=[(X_va_enc, [CLASS_NAMES[i] for i in y_val])],
                categorical_feature=cat_idx,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=60, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
        else:
            model.fit(
                X_tr_enc, [CLASS_NAMES[i] for i in y_train],
                categorical_feature=cat_idx,
                callbacks=[lgb.log_evaluation(period=-1)],
            )
    else:
        model.fit(X_train, y_train)


def predict_proba_ordered(model, X: pd.DataFrame) -> np.ndarray:
    """Return probabilities in [Low, Medium, High] column order."""
    if _HAS_LGB and isinstance(model, lgb.LGBMClassifier):
        X_enc = _encode_cats_for_lgbm(X)
        raw   = model.predict_proba(X_enc)
        # model.classes_ is a list of string labels
        ordered = np.zeros((raw.shape[0], len(CLASS_NAMES)), dtype=raw.dtype)
        for col, label in enumerate(model.classes_):
            ordered[:, LABEL_MAP[label]] = raw[:, col]
        return ordered
    else:
        raw     = model.predict_proba(X)
        ordered = np.zeros((raw.shape[0], len(CLASS_NAMES)), dtype=raw.dtype)
        for col, cls_id in enumerate(model.classes_):
            ordered[:, int(cls_id)] = raw[:, col]
        return ordered


def _get_best_n_iters(model) -> int | None:
    """Return best iteration count for setting final-model n_estimators."""
    if _HAS_LGB and isinstance(model, lgb.LGBMClassifier):
        return getattr(model, "best_iteration_", None)
    return None


# ─── Probability-multiplier tuning ───────────────────────────────────────────

def _grid_search(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, float]:
    best_mults = np.array([1.0, 1.0, 1.0])
    best_score = f1_score(y_true, proba.argmax(axis=1), average="macro")

    # Finer grid than v1 (22 × 37 = 814 combinations)
    med_grid = np.round(np.linspace(0.70, 1.40, 22), 3)
    hi_grid  = np.round(np.linspace(0.20, 3.00, 37), 3)

    for med in med_grid:
        for hi in hi_grid:
            mults = np.array([1.0, med, hi])
            score = f1_score(y_true, (proba * mults).argmax(axis=1), average="macro")
            if score > best_score:
                best_score = score
                best_mults = mults.copy()

    return best_mults, float(best_score)


def tune_probability_multipliers(
    y_true: np.ndarray,
    proba: np.ndarray,
    use_scipy: bool = False,
) -> tuple[np.ndarray, float]:
    grid_mults, grid_score = _grid_search(y_true, proba)

    if not use_scipy or not _HAS_SCIPY:
        return grid_mults, grid_score

    # Nelder-Mead refinement seeded from grid result
    def neg_f1(x: np.ndarray) -> float:
        mults = np.array([1.0, x[0], x[1]])
        return -f1_score(y_true, (proba * mults).argmax(axis=1), average="macro")

    res = minimize(
        neg_f1,
        x0=grid_mults[1:],
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 3000},
    )
    best_mults = np.array([1.0, res.x[0], res.x[1]])
    best_score = -float(res.fun)

    # Fall back to grid if scipy diverged
    if best_score < grid_score:
        return grid_mults, grid_score

    return best_mults, best_score


# ─── Artifact saving ─────────────────────────────────────────────────────────

def save_artifacts(
    output_dir: Path,
    slug: str,
    y_true: np.ndarray,
    default_pred: np.ndarray,
    tuned_pred: np.ndarray,
    multipliers: np.ndarray,
    feature_cols: list[str],
    class_weight: dict[int, float],
    extra: dict | None = None,
) -> None:
    default_f1 = f1_score(y_true, default_pred, average="macro")
    tuned_f1   = f1_score(y_true, tuned_pred,   average="macro")

    metrics: dict = {
        "accuracy_default":  accuracy_score(y_true, default_pred),
        "macro_f1_default":  default_f1,
        "accuracy_tuned":    accuracy_score(y_true, tuned_pred),
        "macro_f1_tuned":    tuned_f1,
        "probability_multipliers": {
            label: float(multipliers[idx]) for idx, label in enumerate(CLASS_NAMES)
        },
        "class_weight": {INV_MAP[idx]: w for idx, w in class_weight.items()},
        "feature_count": len(feature_cols),
        "features":      feature_cols,
    }
    if extra:
        metrics.update(extra)

    (output_dir / "metrics" / f"{slug}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    report = classification_report(
        y_true, tuned_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(output_dir / "metrics" / f"{slug}_classification_report.csv")

    cm = confusion_matrix(y_true, tuned_pred, labels=[0, 1, 2])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "metrics" / f"{slug}_confusion_matrix.csv"
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{slug}\nMacro F1 = {tuned_f1:.4f}")
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / f"{slug}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"\n[{slug}]")
    print(f"  Default macro F1 : {default_f1:.5f}")
    print(f"  Tuned macro F1   : {tuned_f1:.5f}")
    print("  Multipliers      :",
          {label: round(float(multipliers[i]), 4) for i, label in enumerate(CLASS_NAMES)})
    print(classification_report(
        y_true, tuned_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0
    ))


# ─── Blend helper ────────────────────────────────────────────────────────────

def _blend_train_predict(
    args: argparse.Namespace,
    class_weight: dict[int, float],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    X_val_for_es: pd.DataFrame | None = None,
    y_val_for_es: np.ndarray | None = None,
) -> tuple[np.ndarray, HistGradientBoostingClassifier, "lgb.LGBMClassifier | None"]:
    """Train HistGBM + LightGBM on X_train, return averaged probas on X_eval."""
    mh = build_histgbm(args, class_weight)
    fit_model(mh, X_train, y_train)
    ph = predict_proba_ordered(mh, X_eval)

    if not _HAS_LGB:
        warnings.warn("LightGBM unavailable — blend uses HistGBM only.")
        return ph, mh, None

    ml = build_lgbm(args, class_weight)
    fit_model(ml, X_train, y_train, X_val_for_es, y_val_for_es)
    pl = predict_proba_ordered(ml, X_eval)

    return (ph + pl) / 2.0, mh, ml


# ─── Holdout workflow ─────────────────────────────────────────────────────────

def run_holdout(
    args: argparse.Namespace,
    output_dir: Path,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    class_weight: dict[int, float],
    submission: pd.DataFrame,
) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.valid_size, stratify=y, random_state=SEED
    )

    if args.model == "blend":
        valid_proba, _mh, _ml = _blend_train_predict(
            args, class_weight, X_train, y_train, X_valid, X_valid, y_valid
        )
    else:
        model = (build_lgbm if (args.model == "lgbm" and _HAS_LGB) else build_histgbm)(
            args, class_weight
        )
        fit_model(model, X_train, y_train, X_valid, y_valid)
        valid_proba = predict_proba_ordered(model, X_valid)
        best_n = _get_best_n_iters(model)

    multipliers, _ = tune_probability_multipliers(
        y_valid, valid_proba, use_scipy=args.use_scipy_threshold
    )
    tuned_pred   = (valid_proba * multipliers).argmax(axis=1)
    default_pred = valid_proba.argmax(axis=1)

    save_artifacts(
        output_dir, "improved_holdout", y_valid, default_pred, tuned_pred,
        multipliers, feature_cols, class_weight,
    )

    if args.skip_submission:
        print("Skipped submission generation.")
        return

    print("\nTraining final model on full data …")
    if args.model == "blend":
        final_h = build_histgbm(args, class_weight)
        fit_model(final_h, X, y)
        test_ph = predict_proba_ordered(final_h, X_test)

        if _HAS_LGB:
            # Use the best_iteration_ from holdout LightGBM to avoid overfitting
            n_est = int((_ml.best_iteration_ or args.lgb_n_estimators) * 1.10)  # type: ignore[union-attr]
            final_l = build_lgbm(args, class_weight)
            final_l.set_params(n_estimators=n_est)
            fit_model(final_l, X, y)
            test_pl = predict_proba_ordered(final_l, X_test)
            test_proba = (test_ph + test_pl) / 2.0
        else:
            test_proba = test_ph
    else:
        # Reuse best_n from early stopping to set iterations for final model
        final_model = (build_lgbm if (args.model == "lgbm" and _HAS_LGB) else build_histgbm)(
            args, class_weight
        )
        if args.model == "lgbm" and _HAS_LGB and best_n:  # type: ignore[possibly-undefined]
            final_model.set_params(n_estimators=int(best_n * 1.10))
        fit_model(final_model, X, y)
        test_proba = predict_proba_ordered(final_model, X_test)

    test_pred = (test_proba * multipliers).argmax(axis=1)
    submission[TARGET] = [INV_MAP[int(lbl)] for lbl in test_pred]

    sub_path = output_dir / "submissions" / "improved_submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Saved: {sub_path}")
    print(submission[TARGET].value_counts().to_string())


# ─── OOF ensemble workflow ────────────────────────────────────────────────────

def run_oof_ensemble(
    args: argparse.Namespace,
    output_dir: Path,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    class_weight: dict[int, float],
    submission: pd.DataFrame,
) -> None:
    # Reserve a calibration split so multiplier tuning isn't done on the OOF
    # pool itself (which has calibration mismatch across folds).
    if args.oof_tune_on_oof:
        X_oof, y_oof = X, y
        X_calib, y_calib = X, y          # v1 behaviour — tune on full OOF
    else:
        X_oof, X_calib, y_oof, y_calib = train_test_split(
            X, y, test_size=args.oof_calib_size, stratify=y, random_state=SEED
        )

    skf           = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof_proba     = np.zeros((len(y_oof),  len(CLASS_NAMES)), dtype=float)
    calib_proba   = np.zeros((len(X_calib), len(CLASS_NAMES)), dtype=float)
    test_proba_sum = np.zeros((len(X_test), len(CLASS_NAMES)), dtype=float)
    fold_scores: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_oof, y_oof), start=1):
        print(f"\nFold {fold}/{args.folds} …")

        if args.model == "blend":
            fold_val_proba, _mh, _ml = _blend_train_predict(
                args, class_weight,
                X_oof.iloc[tr_idx], y_oof[tr_idx],
                X_oof.iloc[va_idx],
                X_oof.iloc[va_idx], y_oof[va_idx],
            )
        else:
            model = (build_lgbm if (args.model == "lgbm" and _HAS_LGB) else build_histgbm)(
                args, class_weight
            )
            fit_model(
                model,
                X_oof.iloc[tr_idx], y_oof[tr_idx],
                X_oof.iloc[va_idx],  y_oof[va_idx],
            )
            fold_val_proba = predict_proba_ordered(model, X_oof.iloc[va_idx])

        oof_proba[va_idx] = fold_val_proba

        fold_pred = fold_val_proba.argmax(axis=1)
        fs = {
            "fold":     fold,
            "accuracy": accuracy_score(y_oof[va_idx], fold_pred),
            "macro_f1": f1_score(y_oof[va_idx], fold_pred, average="macro"),
        }
        fold_scores.append(fs)
        print(f"  Acc={fs['accuracy']:.5f}  MacroF1={fs['macro_f1']:.5f}")

        if not args.skip_submission:
            test_proba_sum += predict_proba_ordered(
                _mh if args.model == "blend" else model,  # type: ignore[possibly-undefined]
                X_test,
            )
            if args.model == "blend" and _HAS_LGB and _ml is not None:  # type: ignore[possibly-undefined]
                test_proba_sum += predict_proba_ordered(_ml, X_test)  # type: ignore[possibly-undefined]

        # Accumulate calibration-set probabilities from each fold model
        if not args.oof_tune_on_oof:
            if args.model == "blend":
                cp_h = predict_proba_ordered(_mh, X_calib)    # type: ignore[possibly-undefined]
                cp   = cp_h
                if _HAS_LGB and _ml is not None:               # type: ignore[possibly-undefined]
                    cp = (cp_h + predict_proba_ordered(_ml, X_calib)) / 2.0  # type: ignore[possibly-undefined]
                calib_proba += cp
            else:
                calib_proba += predict_proba_ordered(model, X_calib)  # type: ignore[possibly-undefined]

    # Decide where to tune multipliers
    if args.oof_tune_on_oof:
        tune_proba, tune_y = oof_proba, y_oof
    else:
        tune_proba = calib_proba / args.folds   # averaged across folds — same calibration as test
        tune_y     = y_calib

    multipliers, _ = tune_probability_multipliers(
        tune_y, tune_proba, use_scipy=args.use_scipy_threshold
    )
    tuned_pred   = (oof_proba * multipliers).argmax(axis=1)
    default_pred = oof_proba.argmax(axis=1)

    save_artifacts(
        output_dir, "oof_ensemble", y_oof, default_pred, tuned_pred,
        multipliers, feature_cols, class_weight,
        extra={"fold_scores": fold_scores},
    )

    if args.skip_submission:
        print("Skipped submission generation.")
        return

    # For blend mode, test_proba_sum accumulated HistGBM + LightGBM per fold
    divisor = args.folds * (2 if (args.model == "blend" and _HAS_LGB) else 1)
    test_proba = test_proba_sum / divisor
    test_pred  = (test_proba * multipliers).argmax(axis=1)
    submission[TARGET] = [INV_MAP[int(lbl)] for lbl in test_pred]

    sub_path = output_dir / "submissions" / "oof_ensemble_submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\nSaved: {sub_path}")
    print(submission[TARGET].value_counts().to_string())


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    for folder in ("metrics", "figures", "submissions"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    train      = pd.read_csv(data_dir / "train.csv", nrows=args.sample_rows)
    test       = pd.read_csv(data_dir / "test.csv")
    submission = pd.read_csv(data_dir / "sample_submission.csv")

    X, X_test, y, feature_cols = make_feature_matrices(train, test)
    class_weight = make_class_weight(y, args.high_weight_multiplier)

    print(f"\nModel     : {args.model}  (LightGBM available: {_HAS_LGB})")
    print(f"Features  : {len(feature_cols)}")
    print(f"Train rows: {len(y)}")
    print("Class weights:",
          {CLASS_NAMES[k]: round(v, 3) for k, v in class_weight.items()})

    if args.holdout_only:
        run_holdout(args, output_dir, X, X_test, y, feature_cols, class_weight, submission)
    else:
        run_oof_ensemble(args, output_dir, X, X_test, y, feature_cols, class_weight, submission)


if __name__ == "__main__":
    main()
