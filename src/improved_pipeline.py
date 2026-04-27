"""Improved irrigation-need model pipeline.

This script is a stronger, reproducible alternative to the baseline notebook.
It uses sklearn's HistGradientBoostingClassifier because it is available in the
local environment and supports categorical pandas columns directly.

Outputs:
  - outputs/metrics/improved_holdout_metrics.json
  - outputs/metrics/oof_ensemble_metrics.json
  - outputs/metrics/improved_classification_report.csv
  - outputs/metrics/oof_ensemble_classification_report.csv
  - outputs/metrics/improved_confusion_matrix.csv
  - outputs/metrics/oof_ensemble_confusion_matrix.csv
  - outputs/figures/improved_confusion_matrix.png
  - outputs/figures/oof_ensemble_confusion_matrix.png
  - outputs/submissions/improved_submission.csv
  - outputs/submissions/oof_ensemble_submission.csv
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
TARGET = "Irrigation_Need"
CLASS_NAMES = ["Low", "Medium", "High"]
LABEL_MAP = {label: idx for idx, label in enumerate(CLASS_NAMES)}
INV_MAP = {idx: label for label, idx in LABEL_MAP.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an improved irrigation-need classifier."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing Kaggle CSV files.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for metrics and submissions.")
    parser.add_argument("--valid-size", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument("--folds", type=int, default=5, help="Number of OOF folds for ensemble mode.")
    parser.add_argument("--max-iter", type=int, default=350, help="Maximum boosting iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.06, help="Boosting learning rate.")
    parser.add_argument("--high-weight-multiplier", type=float, default=1.0, help="Extra class-weight multiplier for High.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row limit for quick smoke tests.")
    parser.add_argument("--skip-submission", action="store_true", help="Skip final fit and submission generation.")
    parser.add_argument("--holdout-only", action="store_true", help="Run the older single-split workflow.")
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> None:
    for folder in ["metrics", "figures", "submissions"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact domain-inspired features for irrigation stress."""
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
    """Give train and test identical category vocabularies."""
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
    """Tune simple per-class probability multipliers for macro F1."""
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


def save_validation_artifacts(
    output_dir: Path,
    y_valid: np.ndarray,
    default_pred: np.ndarray,
    tuned_pred: np.ndarray,
    multipliers: np.ndarray,
    feature_cols: list[str],
    class_weight: dict[int, float],
) -> None:
    default_macro_f1 = f1_score(y_valid, default_pred, average="macro")
    tuned_macro_f1 = f1_score(y_valid, tuned_pred, average="macro")

    metrics = {
        "validation_accuracy_default": accuracy_score(y_valid, default_pred),
        "validation_macro_f1_default": default_macro_f1,
        "validation_accuracy_tuned": accuracy_score(y_valid, tuned_pred),
        "validation_macro_f1_tuned": tuned_macro_f1,
        "probability_multipliers": {
            label: float(multipliers[idx]) for idx, label in enumerate(CLASS_NAMES)
        },
        "class_weight": {INV_MAP[idx]: weight for idx, weight in class_weight.items()},
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }

    metrics_path = output_dir / "metrics" / "improved_holdout_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(
        y_valid,
        tuned_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "metrics" / "improved_classification_report.csv")

    cm = confusion_matrix(y_valid, tuned_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(output_dir / "metrics" / "improved_confusion_matrix.csv")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Improved HistGradientBoosting\nMacro F1 = {tuned_macro_f1:.4f}")
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "improved_confusion_matrix.png", dpi=160)
    plt.close(fig)


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

    save_validation_artifacts(
        output_dir=output_dir,
        y_valid=y_valid,
        default_pred=default_pred,
        tuned_pred=tuned_pred,
        multipliers=multipliers,
        feature_cols=feature_cols,
        class_weight=class_weight,
    )

    print("Validation results")
    print(f"  Default macro F1: {f1_score(y_valid, default_pred, average='macro'):.5f}")
    print(f"  Tuned macro F1:   {f1_score(y_valid, tuned_pred, average='macro'):.5f}")
    print(
        "  Multipliers:      "
        + ", ".join(f"{label}={multipliers[idx]:.3f}" for idx, label in enumerate(CLASS_NAMES))
    )

    if args.skip_submission:
        print("Skipped final submission generation.")
        return

    final_model = build_model(args, class_weight)
    final_model.fit(X, y)
    test_proba = proba_in_label_order(final_model, X_test)
    test_pred = (test_proba * multipliers).argmax(axis=1)
    submission[TARGET] = [INV_MAP[int(label)] for label in test_pred]

    submission_path = output_dir / "submissions" / "improved_submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved submission: {submission_path}")
    print("Prediction distribution")
    print(submission[TARGET].value_counts().to_string())


def save_oof_artifacts(
    output_dir: Path,
    y: np.ndarray,
    oof_proba: np.ndarray,
    tuned_pred: np.ndarray,
    multipliers: np.ndarray,
    fold_scores: list[dict[str, float]],
    feature_cols: list[str],
    class_weight: dict[int, float],
) -> None:
    default_pred = oof_proba.argmax(axis=1)
    metrics = {
        "oof_accuracy_default": accuracy_score(y, default_pred),
        "oof_macro_f1_default": f1_score(y, default_pred, average="macro"),
        "oof_accuracy_tuned": accuracy_score(y, tuned_pred),
        "oof_macro_f1_tuned": f1_score(y, tuned_pred, average="macro"),
        "probability_multipliers": {
            label: float(multipliers[idx]) for idx, label in enumerate(CLASS_NAMES)
        },
        "class_weight": {INV_MAP[idx]: weight for idx, weight in class_weight.items()},
        "fold_scores": fold_scores,
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }

    metrics_path = output_dir / "metrics" / "oof_ensemble_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(
        y,
        tuned_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / "metrics" / "oof_ensemble_classification_report.csv"
    )

    cm = confusion_matrix(y, tuned_pred, labels=[0, 1, 2])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "metrics" / "oof_ensemble_confusion_matrix.csv"
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"5-Fold OOF HistGradientBoosting\nMacro F1 = {metrics['oof_macro_f1_tuned']:.4f}")
    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "oof_ensemble_confusion_matrix.png", dpi=160)
    plt.close(fig)


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
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof_proba = np.zeros((len(y), len(CLASS_NAMES)), dtype=float)
    test_proba_sum = np.zeros((len(X_test), len(CLASS_NAMES)), dtype=float)
    fold_scores: list[dict[str, float]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Training fold {fold_idx}/{args.folds} ...")
        model = build_model(args, class_weight)
        model.fit(X.iloc[train_idx], y[train_idx])

        valid_proba = proba_in_label_order(model, X.iloc[valid_idx])
        oof_proba[valid_idx] = valid_proba

        valid_pred = valid_proba.argmax(axis=1)
        fold_score = {
            "fold": fold_idx,
            "accuracy": accuracy_score(y[valid_idx], valid_pred),
            "macro_f1": f1_score(y[valid_idx], valid_pred, average="macro"),
        }
        fold_scores.append(fold_score)
        print(
            f"  Fold {fold_idx}: Acc={fold_score['accuracy']:.5f} "
            f"MacroF1={fold_score['macro_f1']:.5f}"
        )

        if not args.skip_submission:
            test_proba_sum += proba_in_label_order(model, X_test)

    multipliers, tuned_macro_f1 = tune_probability_multipliers(y, oof_proba)
    tuned_pred = (oof_proba * multipliers).argmax(axis=1)

    save_oof_artifacts(
        output_dir=output_dir,
        y=y,
        oof_proba=oof_proba,
        tuned_pred=tuned_pred,
        multipliers=multipliers,
        fold_scores=fold_scores,
        feature_cols=feature_cols,
        class_weight=class_weight,
    )

    print("OOF ensemble results")
    print(f"  Default macro F1: {f1_score(y, oof_proba.argmax(axis=1), average='macro'):.5f}")
    print(f"  Tuned macro F1:   {tuned_macro_f1:.5f}")
    print(
        "  Multipliers:      "
        + ", ".join(f"{label}={multipliers[idx]:.3f}" for idx, label in enumerate(CLASS_NAMES))
    )

    if args.skip_submission:
        print("Skipped ensemble submission generation.")
        return

    test_proba = test_proba_sum / args.folds
    test_pred = (test_proba * multipliers).argmax(axis=1)
    submission[TARGET] = [INV_MAP[int(label)] for label in test_pred]

    submission_path = output_dir / "submissions" / "oof_ensemble_submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved submission: {submission_path}")
    print("Prediction distribution")
    print(submission[TARGET].value_counts().to_string())


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dirs(output_dir)

    train = pd.read_csv(data_dir / "train.csv", nrows=args.sample_rows)
    test = pd.read_csv(data_dir / "test.csv")
    submission = pd.read_csv(data_dir / "sample_submission.csv")

    X, X_test, y, feature_cols = make_feature_matrices(train, test)
    class_weight = make_class_weight(y, args.high_weight_multiplier)

    if args.holdout_only:
        run_holdout(args, output_dir, X, X_test, y, feature_cols, class_weight, submission)
    else:
        run_oof_ensemble(args, output_dir, X, X_test, y, feature_cols, class_weight, submission)


if __name__ == "__main__":
    main()
