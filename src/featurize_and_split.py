"""
DrugSafe — Day 2: Featurization & Data Splitting
==================================================

Input:  data/processed/dilirank_processed.csv   (from Day 1)
Output: data/processed/features.npz              (fingerprints + descriptors)
        data/processed/splits.csv                (train/test labels for both split strategies)
        figures/scaffold_split_summary.png

Run:    python src/featurize_and_split.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("drugsafe.featurize_and_split")

PROCESSED_PATH = Path("data/processed/dilirank_processed.csv")
FEATURES_PATH = Path("data/processed/features.npz")
SPLITS_PATH = Path("data/processed/splits.csv")
FIGURES_DIR = Path("figures")

SMILES_COL = "smiles_std"
LABEL_COL = "dili_label"

RANDOM_SEED = 42
TEST_FRACTION = 0.20

FP_RADIUS = 2
FP_NBITS = 2048


# ---------------------------------------------------------------------------
# Step 1: Featurization — fingerprints + physicochemical descriptors
# ---------------------------------------------------------------------------
def compute_morgan_fingerprint(mol) -> np.ndarray:
    """
    A Morgan fingerprint is a long 0/1 vector. Each bit roughly answers:
    'is this specific small chemical pattern present in the molecule?'
    Two molecules that share a lot of 1s in the same positions are
    structurally similar.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
    arr = np.zeros((FP_NBITS,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_descriptors(mol) -> dict:
    """
    Plain physicochemical properties. These are simple numbers, not
    patterns — e.g. molecular weight, how 'oily vs watery' it is (LogP),
    how many hydrogen-bond donors/acceptors it has, etc.
    """
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }


def featurize_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
    fingerprints = []
    descriptor_rows = []
    valid_idx = []

    for i, smi in enumerate(df[SMILES_COL]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log.warning(f"Row {i}: SMILES failed to parse at featurization time, skipping: {smi}")
            continue
        fingerprints.append(compute_morgan_fingerprint(mol))
        descriptor_rows.append(compute_descriptors(mol))
        valid_idx.append(i)

    fp_matrix = np.vstack(fingerprints)
    descriptor_df = pd.DataFrame(descriptor_rows)
    log.info(f"Featurized {len(valid_idx)}/{len(df)} molecules "
              f"({fp_matrix.shape[1]}-bit fingerprint + {descriptor_df.shape[1]} descriptors)")
    return fp_matrix, descriptor_df, valid_idx


# ---------------------------------------------------------------------------
# Step 2a: Random stratified split
# ---------------------------------------------------------------------------
def random_stratified_split(df: pd.DataFrame) -> pd.Series:
    """
    Simple, standard split: randomly assign 80% to train, 20% to test,
    keeping the same class balance (concern vs no-concern) in both piles.
    Easy, but structurally similar drugs can still end up split across
    train and test — that's what the scaffold split below checks for.
    """
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_FRACTION,
        stratify=df[LABEL_COL],
        random_state=RANDOM_SEED,
    )
    split_col = pd.Series("train", index=df.index)
    split_col.loc[test_idx] = "test"
    log.info(f"Random split: {(split_col == 'train').sum()} train / {(split_col == 'test').sum()} test")
    return split_col


# ---------------------------------------------------------------------------
# Step 2b: Scaffold split (Bemis-Murcko)
# ---------------------------------------------------------------------------
def get_scaffold(smiles: str) -> str:
    """
    A 'scaffold' is a molecule's core skeleton with side chains chopped off —
    two drugs built on the same skeleton (e.g. two different painkillers
    from the same chemical family) get the same scaffold string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(df: pd.DataFrame) -> pd.Series:
    """
    Groups molecules by scaffold, then assigns whole scaffold groups to
    train or test — never splitting one scaffold family across both.
    This is a harder, more honest test: the model can't just recognize
    'I've seen this skeleton before' to do well.

    We fill test with the largest scaffold groups first, then keep adding
    smaller groups until we're close to the target test fraction. This
    keeps the split reasonably balanced without needing many small groups
    scattered awkwardly.
    """
    scaffolds = df[SMILES_COL].apply(get_scaffold)
    scaffold_groups = scaffolds.groupby(scaffolds).groups  # scaffold -> index list

    groups_sorted = sorted(scaffold_groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    target_test_size = int(len(df) * TEST_FRACTION)
    test_indices = []
    train_indices = []

    for scaffold, idx_list in groups_sorted:
        if len(test_indices) < target_test_size:
            test_indices.extend(idx_list)
        else:
            train_indices.extend(idx_list)

    split_col = pd.Series("train", index=df.index)
    split_col.loc[test_indices] = "test"

    n_scaffolds = len(scaffold_groups)
    log.info(f"Scaffold split: {n_scaffolds} unique scaffolds found")
    log.info(f"Scaffold split: {(split_col == 'train').sum()} train / {(split_col == 'test').sum()} test")

    train_balance = df.loc[split_col == "train", LABEL_COL].value_counts(normalize=True)
    test_balance = df.loc[split_col == "test", LABEL_COL].value_counts(normalize=True)
    log.info(f"Scaffold split class balance — train: {train_balance.to_dict()}, test: {test_balance.to_dict()}")

    return split_col


def plot_scaffold_split_summary(df: pd.DataFrame, scaffold_col: pd.Series, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, split_name in zip(axes, ["train", "test"]):
        subset = df.loc[scaffold_col == split_name, LABEL_COL]
        counts = subset.value_counts().sort_index()
        ax.bar(["No Concern", "Concern"], counts.reindex([0, 1], fill_value=0).values,
               color=["#4C72B0", "#C44E52"])
        ax.set_title(f"Scaffold split — {split_name} (n={len(subset)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info(f"Saved scaffold split summary plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(PROCESSED_PATH)
    log.info(f"Loaded {len(df)} processed compounds from {PROCESSED_PATH}")

    fp_matrix, descriptor_df, valid_idx = featurize_dataframe(df)
    df = df.loc[valid_idx].reset_index(drop=True)
    descriptor_df = descriptor_df.reset_index(drop=True)

    np.savez_compressed(
        FEATURES_PATH,
        fingerprints=fp_matrix,
        descriptors=descriptor_df.values,
        descriptor_names=descriptor_df.columns.to_numpy(),
        labels=df[LABEL_COL].to_numpy(),
    )
    log.info(f"Saved features to {FEATURES_PATH}")

    df["split_random"] = random_stratified_split(df)
    df["split_scaffold"] = scaffold_split(df)

    df[["CompoundName", SMILES_COL, LABEL_COL, "split_random", "split_scaffold"]].to_csv(
        SPLITS_PATH, index=False
    )
    log.info(f"Saved split assignments to {SPLITS_PATH}")

    plot_scaffold_split_summary(df, df["split_scaffold"], FIGURES_DIR / "scaffold_split_summary.png")


if __name__ == "__main__":
    main()
