"""
DrugSafe — Day 3: Model Training & Evaluation
================================================

Input:  data/processed/features.npz    (from Day 2)
        data/processed/splits.csv       (from Day 2)
Output: data/processed/model_metrics.csv
        models/*.joblib                 (trained models, one per model x split)
        figures/roc_pr_split_random.png
        figures/roc_pr_split_scaffold.png
        figures/confusion_matrices.png

Run:    python src/train_and_evaluate.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("drugsafe.train_and_evaluate")

FEATURES_PATH = Path("data/processed/features.npz")
SPLITS_PATH = Path("data/processed/splits.csv")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("figures")
METRICS_PATH = Path("data/processed/model_metrics.csv")

RANDOM_SEED = 42
SPLIT_COLUMNS = ["split_random", "split_scaffold"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    data = np.load(FEATURES_PATH, allow_pickle=True)
    # Combine the fingerprint (structural pattern) and descriptor (simple
    # property) features into one matrix — the model trains on both together.
    X = np.hstack([data["fingerprints"], data["descriptors"]])
    y = data["labels"]
    splits_df = pd.read_csv(SPLITS_PATH)
    log.info(f"Loaded feature matrix: {X.shape[0]} molecules x {X.shape[1]} features")
    return X, y, splits_df


def get_split_masks(splits_df: pd.DataFrame, split_col: str):
    train_mask = (splits_df[split_col] == "train").to_numpy()
    test_mask = (splits_df[split_col] == "test").to_numpy()
    return train_mask, test_mask


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_models():
    """
    class_weight='balanced' tells both models to pay extra attention to
    the minority class instead of just favoring whichever class has more
    examples — important since our data is mildly imbalanced (~60/40).
    """
    logistic_regression = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    random_forest = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return {"LogisticRegression": logistic_regression, "RandomForest": random_forest}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall_sensitivity": recall_score(y_true, y_pred),
        "specificity": specificity,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# ---------------------------------------------------------------------------
# Train + evaluate one split
# ---------------------------------------------------------------------------
def train_and_evaluate_split(X, y, splits_df, split_col: str):
    train_mask, test_mask = get_split_masks(splits_df, split_col)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    log.info(f"[{split_col}] train={len(y_train)} test={len(y_test)}")

    models = build_models()  # fresh, untrained models for this split
    results = {}
    curves = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        results[name] = metrics

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        curves[name] = {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
                         "y_test": y_test, "y_pred": y_pred}

        model_path = MODELS_DIR / f"{name}_{split_col}.joblib"
        joblib.dump(model, model_path)

        log.info(
            f"[{split_col}] {name}: ROC-AUC={metrics['roc_auc']:.3f} "
            f"PR-AUC={metrics['pr_auc']:.3f} F1={metrics['f1']:.3f} "
            f"MCC={metrics['mcc']:.3f} Specificity={metrics['specificity']:.3f}"
        )

    return results, curves


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc_pr(curves: dict, split_col: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for name, c in curves.items():
        axes[0].plot(c["fpr"], c["tpr"], label=name)
        axes[1].plot(c["rec"], c["prec"], label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random guess")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — {split_col}")
    axes[0].legend()
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {split_col}")
    axes[1].legend()
    fig.tight_layout()
    out_path = FIGURES_DIR / f"roc_pr_{split_col}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved ROC/PR curves to {out_path}")


def plot_confusion_matrices(all_curves: dict):
    """One grid: rows = split type, columns = model."""
    n_rows = len(all_curves)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, (split_col, curves) in enumerate(all_curves.items()):
        for col, (name, c) in enumerate(curves.items()):
            cm = confusion_matrix(c["y_test"], c["y_pred"])
            ax = axes[row, col]
            im = ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["No Concern", "Concern"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["No Concern", "Concern"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"{name} — {split_col}")

    fig.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrices.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved confusion matrix grid to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y, splits_df = load_data()

    all_results = []
    all_curves = {}
    for split_col in SPLIT_COLUMNS:
        results, curves = train_and_evaluate_split(X, y, splits_df, split_col)
        plot_roc_pr(curves, split_col)
        all_curves[split_col] = curves
        for name, metrics in results.items():
            all_results.append({"split": split_col, "model": name, **metrics})

    plot_confusion_matrices(all_curves)

    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(METRICS_PATH, index=False)
    log.info(f"Saved full metrics table to {METRICS_PATH}")
    log.info("\n" + metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
