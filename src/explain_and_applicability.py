"""
DrugSafe — Day 4: Explainability & Applicability Domain
==========================================================

Input:  data/processed/dilirank_processed.csv
        data/processed/features.npz
        data/processed/splits.csv
        models/RandomForest_split_random.joblib   (from Day 3)
Output: figures/shap_feature_importance.png
        figures/top_bit_substructures.png
        Prints applicability-domain demo results to the console

Run:    python src/explain_and_applicability.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("drugsafe.explain")

PROCESSED_PATH = Path("data/processed/dilirank_processed.csv")
FEATURES_PATH = Path("data/processed/features.npz")
SPLITS_PATH = Path("data/processed/splits.csv")
MODEL_PATH = Path("models/RandomForest_split_random.joblib")
FIGURES_DIR = Path("figures")

FP_RADIUS = 2
FP_NBITS = 2048
DESCRIPTOR_NAMES = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "RingCount", "AromaticRings"]
TOP_N_FEATURES = 15
TOP_N_SUBSTRUCTURES = 6


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
def load_everything():
    df = pd.read_csv(PROCESSED_PATH)
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X = np.hstack([data["fingerprints"], data["descriptors"]])
    splits_df = pd.read_csv(SPLITS_PATH)
    model = joblib.load(MODEL_PATH)
    log.info(f"Loaded {len(df)} compounds, feature matrix {X.shape}, model from {MODEL_PATH}")
    return df, X, splits_df, model


def get_feature_names():
    return [f"fp_bit_{i}" for i in range(FP_NBITS)] + DESCRIPTOR_NAMES


# ---------------------------------------------------------------------------
# SHAP: which features matter, on average, for the model's predictions
# ---------------------------------------------------------------------------
def compute_shap_importance(model, X_test, feature_names):
    """
    TreeExplainer is fast and exact for tree-based models like Random Forest
    (no approximation needed, unlike SHAP on arbitrary black-box models).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    # Different SHAP versions return either a list [class0_values, class1_values]
    # or a single array for binary classifiers — handle both.
    if isinstance(shap_values, list):
        sv = shap_values[1]  # class 1 = DILI Concern
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, out_path: Path):
    top = importance_df.head(TOP_N_FEATURES).iloc[::-1]  # reverse so #1 is at the top of the chart
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| — average impact on the model's prediction")
    ax.set_title("What drives DrugSafe's DILI-concern predictions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved feature importance chart to {out_path}")


# ---------------------------------------------------------------------------
# Connect important fingerprint bits back to actual chemical substructures
# ---------------------------------------------------------------------------
def find_example_for_bit(df: pd.DataFrame, bit_id: int):
    """
    A fingerprint bit doesn't 'mean' anything on its own — it's just a slot
    that lights up (1) when a specific small chemical pattern is present.
    To show *which* pattern, we scan for one training molecule that has this
    bit switched on and ask RDKit which atoms caused it.
    """
    for smi in df["smiles_std"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS, bitInfo=bit_info)
        if bit_id in bit_info:
            return mol, bit_info
    return None, None


def render_top_bit_substructures(df: pd.DataFrame, importance_df: pd.DataFrame, out_path: Path):
    bit_features = importance_df[importance_df["feature"].str.startswith("fp_bit_")]

    tuples, legends = [], []
    for feat in bit_features["feature"].head(TOP_N_SUBSTRUCTURES):
        bit_id = int(feat.split("_")[-1])
        mol, bit_info = find_example_for_bit(df, bit_id)
        if mol is None:
            log.warning(f"Could not find an example molecule for bit {bit_id}, skipping")
            continue
        tuples.append((mol, bit_id, bit_info))
        legends.append(f"bit {bit_id}")

    if not tuples:
        log.warning("No substructures could be rendered — skipping image")
        return

    img = Draw.DrawMorganBits(tuples, molsPerRow=3, legends=legends, useSVG=False)

    # RDKit versions differ in what this returns: a PIL Image (.save()),
    # raw SVG text (str), or raw PNG bytes (bytes). Handle all three so the
    # script works regardless of which RDKit version is installed.
    if hasattr(img, "save"):
        img.save(str(out_path))
    elif isinstance(img, bytes):
        with open(out_path, "wb") as f:
            f.write(img)
    elif isinstance(img, str):
        out_path = out_path.with_suffix(".svg")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(img)
    else:
        log.warning(f"Unrecognized image return type ({type(img)}) — could not save substructure image")
        return

    log.info(f"Saved top substructure images to {out_path}")


# ---------------------------------------------------------------------------
# Applicability domain: how similar is a query molecule to the training data?
# ---------------------------------------------------------------------------
def build_reference_fingerprints(df: pd.DataFrame, train_mask: np.ndarray):
    """
    IMPORTANT: only training-set molecules go into this reference — using
    test-set molecules here would leak test information into the
    'how familiar is this' check, defeating its purpose.
    """
    fps = []
    for smi in df.loc[train_mask, "smiles_std"]:
        mol = Chem.MolFromSmiles(smi)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS))
    return fps


def similarity_tier(query_smiles: str, reference_fps: list, k: int = 5) -> dict | None:
    """
    Returns a similarity score (0-1, higher = more familiar) and a plain-
    language tier. This is a heuristic proxy, not a formally validated
    applicability domain — say so wherever it's shown to the user.
    """
    mol = Chem.MolFromSmiles(query_smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
    sims = sorted(DataStructs.BulkTanimotoSimilarity(fp, reference_fps), reverse=True)
    max_sim = sims[0]
    mean_top_k = sum(sims[:k]) / k

    if mean_top_k >= 0.5:
        tier = "High"
    elif mean_top_k >= 0.3:
        tier = "Medium"
    else:
        tier = "Low"

    return {"max_similarity": round(max_sim, 3), "mean_top5_similarity": round(mean_top_k, 3), "tier": tier}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, X, splits_df, model = load_everything()
    feature_names = get_feature_names()

    test_mask = (splits_df["split_random"] == "test").to_numpy()
    train_mask = (splits_df["split_random"] == "train").to_numpy()
    X_test = X[test_mask]

    log.info("Computing SHAP values (this can take ~10-30 seconds)...")
    importance_df = compute_shap_importance(model, X_test, feature_names)
    log.info(f"Top {TOP_N_FEATURES} features:\n{importance_df.head(TOP_N_FEATURES).to_string(index=False)}")
    plot_feature_importance(importance_df, FIGURES_DIR / "shap_feature_importance.png")

    log.info("Rendering substructures for top fingerprint bits...")
    render_top_bit_substructures(df, importance_df, FIGURES_DIR / "top_bit_substructures.png")

    log.info("Building applicability-domain reference set from training molecules...")
    reference_fps = build_reference_fingerprints(df, train_mask)

    log.info("Applicability-domain demo on a few example molecules:")
    demo_smiles = {
        "Aspirin (common, in-domain)": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen (common, in-domain)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "A deliberately unusual synthetic structure": "C1=CC2=C(C=C1[Si](C)(C)C)N=C(S2)N3CCN(CC3)C4=NC5=CC=CC=C5N4",
    }
    for label, smi in demo_smiles.items():
        result = similarity_tier(smi, reference_fps)
        log.info(f"  {label}: {result}")


if __name__ == "__main__":
    main()
