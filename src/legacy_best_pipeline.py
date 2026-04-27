"""Reproduce the legacy best public submission.

This keeps the older HistGradientBoosting setup separate from the newer
LightGBM/blend experiments. Use it when the newer experiments score worse on
Kaggle and you want to recreate the known-good `improved_submission.csv` style.
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
TARGET = "Irrigation_Need"
CLASS_NAMES = ["Low", "Medium", "High"]
LABEL_MAP = {label: idx for idx, label in enumerate(CLASS_NAMES)}
INV_MAP = {idx: label for label, idx in LABEL_MAP.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the legacy best HistGBM pipeline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--high-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--skip-submission", action="store_true")
    return parser.parse_args()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    area = df["Field_Area_hectare"].clip(lower=0.001)
    humidity = df["Humidity"].clip(lower=0.001)

    df["Rainfall_per_hectare"] = df["Rainfall_mm"] / area
    df["Total_Water_mm"] = df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
    df["Rain_minus_Previous_Irrigation"] = df["Rainfall_mm"] - df["Previous_Irrigation_mm"]
    df["Moisture_Deficit"] = (50.0 - df["Soil_Moisture"]).clip(lower=0.0)
    df["Heat_Dryness_Index"] = df["Temperature_C"] * (100.0 - df["Humidity"]) / 100.0
    df["Evaporation_Proxy"] = (
        df["Temperature_C"] * df["Sunlight_Hours"] * df["Wind_Speed_kmh"] / humidity
    )
    df["Soil_Water_Retention"] = df["Soil_Moisture"] * df["Organic_Carbon"]
    df["Salinity_Stress"] = df["Electrical_Conductivity"] * df["Temperature_C"]
    df["Moisture_Rainfall_Ratio"] = df["Soil_Moisture"] / (df["Rainfall_mm"] + 1.0)

    df["Crop_Season"] = df["Crop_Type"].astype(str) + "_" + df["Season"].astype(str)
    df["Soil_Irrigation"] = df["Soil_Type"].astype(str) + "_" + df["Irrigation_Type"].astype(str)
    df["Region_Water"] = df["Region"].astype(str) + "_" + df["Water_Source"].astype(str)

    return df


def align_categorical_dtypes(
    train: pd.DataFrame, test: pd.DataFrame, categorical_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    for col in categorical_cols:
        values = pd.concat([train[col].astype(str), test[col].astype(str)], ignore_index=True)
        categories = pd.Index(values.unique())
        train[col] = pd.Categorical(train[col].astype(str), categories=categories)
        test[col] = pd.Categorical(test[col].astype(str), categories=categories)

    return train, test


def make_feature_matrices(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    train_fe = add_features(train)
    test_fe = add_features(test)

    categorical_cols = [
        col for col in train_fe.select_dtypes(include=["object", "category"]).columns if col != TARGET
    ]
    train_fe, test_fe = align_categorical_dtypes(train_fe, test_fe, categorical_cols)

    feature_cols = [col for col in train_fe.columns if col not in ["id", TARGET]]
    X = train_fe[feature_cols]
    X_test = test_fe[feature_cols]
    y = train_fe[TARGET].map(LABEL_MAP).to_numpy()

    return X, X_test, y, feature_cols


def make_class_weight(y: np.ndarray, high_multiplier: float) -> dict[int, float]:
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y)
    class_weight = {idx: float(weight) for idx, weight in enumerate(weights)}
    class_weight[LABEL_MAP["High"]] *= high_multiplier
    return class_weight


def build_model(args: argparse.Namespace, class_weight: dict[int, float]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=63,
        min_samples_leaf=35,
        l2_regularization=0.03,
        max_bins=255,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=20,
        random_state=SEED,
        class_weight=class_weight,
    )


def proba_in_label_order(model: HistGradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    raw_proba = model.predict_proba(X)
    ordered = np.zeros((raw_proba.shape[0], len(CLASS_NAMES)), dtype=raw_proba.dtype)
    for model_col, class_id in enumerate(model.classes_):
        ordered[:, int(class_id)] = raw_proba[:, model_col]
    return ordered


def tune_probability_multipliers(y_true: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, float]:
    best_multipliers = np.array([1.0, 1.0, 1.0])
    best_score = f1_score(y_true, proba.argmax(axis=1), average="macro")

    medium_grid = np.round(np.linspace(0.75, 1.35, 13), 3)
    high_grid = np.round(np.linspace(0.35, 2.75, 25), 3)

    for medium_multiplier in medium_grid:
        for high_multiplier in high_grid:
            multipliers = np.array([1.0, medium_multiplier, high_multiplier])
            preds = (proba * multipliers).argmax(axis=1)
            score = f1_score(y_true, preds, average="macro")
            if score > best_score:
                best_score = score
                best_multipliers = multipliers

    return best_multipliers, float(best_score)


def save_artifacts(
    output_dir: Path,
    y_valid: np.ndarray,
    default_pred: np.ndarray,
    tuned_pred: np.ndarray,
    multipliers: np.ndarray,
    feature_cols: list[str],
    class_weight: dict[int, float],
) -> None:
    metrics = {
        "validation_accuracy_default": accuracy_score(y_valid, default_pred),
        "validation_macro_f1_default": f1_score(y_valid, default_pred, average="macro"),
        "validation_accuracy_tuned": accuracy_score(y_valid, tuned_pred),
        "validation_macro_f1_tuned": f1_score(y_valid, tuned_pred, average="macro"),
        "probability_multipliers": {
            label: float(multipliers[idx]) for idx, label in enumerate(CLASS_NAMES)
        },
        "class_weight": {INV_MAP[idx]: weight for idx, weight in class_weight.items()},
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }

    (output_dir / "metrics" / "legacy_best_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    report = classification_report(
        y_valid,
        tuned_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / "metrics" / "legacy_best_classification_report.csv"
    )

    cm = confusion_matrix(y_valid, tuned_pred, labels=[0, 1, 2])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "metrics" / "legacy_best_confusion_matrix.csv"
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Legacy Best HistGBM\nMacro F1 = {metrics['validation_macro_f1_tuned']:.4f}")
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "legacy_best_confusion_matrix.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    for folder in ["metrics", "figures", "submissions"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv", nrows=args.sample_rows)
    test = pd.read_csv(data_dir / "test.csv")
    submission = pd.read_csv(data_dir / "sample_submission.csv")

    X, X_test, y, feature_cols = make_feature_matrices(train, test)
    class_weight = make_class_weight(y, args.high_weight_multiplier)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.valid_size,
        stratify=y,
        random_state=SEED,
    )

    model = build_model(args, class_weight)
    model.fit(X_train, y_train)

    valid_proba = proba_in_label_order(model, X_valid)
    default_pred = valid_proba.argmax(axis=1)
    multipliers, _ = tune_probability_multipliers(y_valid, valid_proba)
    tuned_pred = (valid_proba * multipliers).argmax(axis=1)

    save_artifacts(output_dir, y_valid, default_pred, tuned_pred, multipliers, feature_cols, class_weight)

    print("Legacy best validation results")
    print(f"  Default macro F1: {f1_score(y_valid, default_pred, average='macro'):.5f}")
    print(f"  Tuned macro F1:   {f1_score(y_valid, tuned_pred, average='macro'):.5f}")
    print(
        "  Multipliers:      "
        + ", ".join(f"{label}={multipliers[idx]:.3f}" for idx, label in enumerate(CLASS_NAMES))
    )

    if args.skip_submission:
        print("Skipped submission generation.")
        return

    final_model = build_model(args, class_weight)
    final_model.fit(X, y)
    test_proba = proba_in_label_order(final_model, X_test)
    test_pred = (test_proba * multipliers).argmax(axis=1)
    submission[TARGET] = [INV_MAP[int(label)] for label in test_pred]

    submission_path = output_dir / "submissions" / "legacy_best_submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved: {submission_path}")
    print(submission[TARGET].value_counts().to_string())


if __name__ == "__main__":
    main()
