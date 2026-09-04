"""
DrugSafe — Final model for deployment
========================================
Retrains Random Forest on ALL 847 compounds (not a train/test subset).
The train/test splits were for honestly measuring performance (Day 3) —
now that we know how well it generalizes, the deployed model should learn
from every available compound.

Input:  data/processed/features.npz
Output: models/final_model.joblib

Run:    python src/train_final_model.py
"""

import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("drugsafe.train_final_model")

FEATURES_PATH = Path("data/processed/features.npz")
MODEL_PATH = Path("models/final_model.joblib")
RANDOM_SEED = 42


def main():
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X = np.hstack([data["fingerprints"], data["descriptors"]])
    y = data["labels"]
    log.info(f"Training final model on all {X.shape[0]} compounds, {X.shape[1]} features")

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info(f"Saved final deployment model to {MODEL_PATH}")
    log.info(
        "Note: this model has seen 100% of the dataset, so its own training "
        "accuracy is not a meaningful performance number — refer back to "
        "Day 3's held-out scaffold-split metrics for the honest estimate."
    )


if __name__ == "__main__":
    main()
