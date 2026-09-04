"""
DrugSafe — Day 1: Data Acquisition & Standardization
======================================================

Input:  data/raw/dilirank2.xlsx  (downloaded manually — see step 0 below)
Output: data/processed/dilirank_processed.csv
        figures/class_distribution.png

Run:    python src/data_prep.py
"""

import time
import logging
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # RDKit is noisy about invalid SMILES; we handle it ourselves
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("drugsafe.data_prep")

RAW_PATH = Path("data/raw/dilirank2.xlsx")
RAW_SHEET_NAME = "version 2"   # the "version 1" sheet is the old DILIrank 1.0 snapshot — don't use it
RAW_HEADER_ROW = 1             # 0-indexed: real headers sit on the 2nd row of this sheet
PROCESSED_PATH = Path("data/processed/dilirank_processed.csv")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

# DILIrank 2.0's four categories -> binary target.
# Ambiguous is dropped, per the project design.
#
# NOTE: the raw vDILI-Concern column has inconsistent capitalization in the
# FDA export (e.g. "vMOST-DILI-concern" vs "vMost-DILI-concern", and
# "vNo-DILI-Concern" vs "vNo-DILI-concern" — 3 rows total are affected out of
# 1,336). We normalize to lowercase before mapping so these aren't silently
# dropped as unmapped categories.
LABEL_MAP = {
    "vmost-dili-concern": 1,
    "vless-dili-concern": 1,
    "vno-dili-concern": 0,
    "ambiguous-dili-concern": None,  # dropped
}


# ---------------------------------------------------------------------------
# Step 0: Manual download
# ---------------------------------------------------------------------------
# DILIrank 2.0 isn't behind a stable public API — pull it by hand once:
#   https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/
#   drug-induced-liver-injury-rank-dilirank-20-dataset
# Save the spreadsheet as data/raw/dilirank2.xlsx.
# Open it once yourself first and confirm the actual column names — FDA
# spreadsheets vary release to release. Update COLUMN_MAP below to match.

COLUMN_MAP = {
    "drug_name_col": "CompoundName",
    "label_col": "vDILI-Concern",
}


def load_raw_dilirank() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Download DILIrank 2.0 from the FDA LTKB page "
            "and save it there before running this script."
        )
    df = pd.read_excel(RAW_PATH, sheet_name=RAW_SHEET_NAME, header=RAW_HEADER_ROW)
    log.info(f"Loaded {len(df)} raw entries from {RAW_PATH} (sheet: {RAW_SHEET_NAME})")
    return df


# ---------------------------------------------------------------------------
# Step 1: Resolve drug names -> SMILES via PubChem PUG REST
# ---------------------------------------------------------------------------
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
    "ConnectivitySMILES/JSON"
)


_diagnostic_shown = False  # print one detailed failure reason, not thousands


def fetch_smiles_from_pubchem(drug_name: str, sleep: float = 0.2) -> str | None:
    """Look up a single drug name in PubChem. Returns SMILES or None."""
    global _diagnostic_shown
    # Drug names often contain spaces ("Abacavir sulfate"), which are not
    # valid raw characters in a URL. They must be percent-encoded (e.g.
    # space -> %20) or PubChem rejects the request outright.
    encoded_name = quote(drug_name, safe="")
    url = PUBCHEM_URL.format(name=encoded_name)
    try:
        resp = requests.get(url, timeout=10)
        time.sleep(sleep)  # be polite to PubChem's rate limits
        if resp.status_code != 200:
            if not _diagnostic_shown:
                log.warning(
                    f"Example failed lookup — '{drug_name}' returned HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
                _diagnostic_shown = True
            return None
        props = resp.json()["PropertyTable"]["Properties"][0]
        if "ConnectivitySMILES" not in props:
            if not _diagnostic_shown:
                log.warning(
                    f"PubChem returned HTTP 200 for '{drug_name}' but the response has no "
                    f"'ConnectivitySMILES' field — only {list(props.keys())}. "
                    f"PubChem likely renamed this property again; update PUBCHEM_URL and "
                    f"the props.get(...) call in fetch_smiles_from_pubchem() to match."
                )
                _diagnostic_shown = True
            return None
        return props.get("ConnectivitySMILES")
    except Exception as e:
        if not _diagnostic_shown:
            log.warning(f"Example failed lookup — '{drug_name}' raised an error: {e}")
            _diagnostic_shown = True
        return None


def resolve_smiles(df: pd.DataFrame, name_col: str, cache_path: Path = Path("data/raw/smiles_cache.csv")) -> pd.DataFrame:
    """
    Map each drug name to a SMILES string, with a local cache so re-runs
    don't re-hit PubChem for names you've already resolved.
    """
    cache = {}
    if cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        cache = dict(zip(cached_df["drug_name"], cached_df["smiles"]))
        log.info(f"Loaded {len(cache)} cached SMILES lookups")

    smiles_list = []
    unresolved = []
    for name in df[name_col]:
        if name in cache and pd.notna(cache[name]):
            smiles_list.append(cache[name])
            continue
        smi = fetch_smiles_from_pubchem(name)
        smiles_list.append(smi)
        cache[name] = smi
        if smi is None:
            unresolved.append(name)

    pd.DataFrame({"drug_name": list(cache.keys()), "smiles": list(cache.values())}).to_csv(
        cache_path, index=False
    )

    df = df.copy()
    df["smiles_raw"] = smiles_list

    n_resolved = df["smiles_raw"].notna().sum()
    log.info(f"Resolved {n_resolved}/{len(df)} drug names to SMILES")

    # Sanity check: a near-total resolution failure almost always means the
    # PubChem API changed shape (e.g. a renamed property field) rather than
    # 1,336 drug names genuinely not existing in PubChem. Fail loudly here
    # instead of silently cascading into an empty processed dataset.
    resolution_rate = n_resolved / len(df) if len(df) else 0
    if resolution_rate < 0.5:
        raise RuntimeError(
            f"Only {n_resolved}/{len(df)} ({resolution_rate:.0%}) drug names resolved to "
            "SMILES — this is far below the expected ~70-90% success rate and usually means "
            "the PubChem API response format has changed. Check the warning above for the "
            "actual field names PubChem returned, update fetch_smiles_from_pubchem() to match, "
            "delete data/raw/smiles_cache.csv, and re-run before trusting any downstream output."
        )

    if unresolved:
        log.warning(
            f"{len(unresolved)} names could not be resolved automatically "
            f"(likely biologics, combination products, or name-format mismatches). "
            f"First 10: {unresolved[:10]}"
        )
    return df


# ---------------------------------------------------------------------------
# Step 2: RDKit standardization
# ---------------------------------------------------------------------------
_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize_smiles(smiles: str) -> str | None:
    """
    Parse -> strip salts/solvate fragments -> neutralize charges -> canonicalize.
    Returns None if the SMILES is invalid or fails standardization.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _largest_fragment.choose(mol)   # keep parent compound, drop salt/solvate
        mol = _uncharger.uncharge(mol)        # neutralize where chemically sensible
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["smiles_std"] = df["smiles_raw"].apply(standardize_smiles)
    n_before = df["smiles_raw"].notna().sum()
    n_after = df["smiles_std"].notna().sum()
    log.info(f"Standardization: {n_after}/{n_before} SMILES parsed and standardized successfully")
    df = df[df["smiles_std"].notna()].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 3: Deduplicate on canonical SMILES
# ---------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Different trade names / salt forms can canonicalize to the same structure.
    Where duplicates disagree on label, keep the more severe (DILI concern = 1)
    call and log it — this is a conservative, defensible default, but flag it
    in your report rather than hiding it.
    """
    before = len(df)
    conflicts = (
        df.groupby("smiles_std")[label_col]
        .nunique()
        .loc[lambda s: s > 1]
    )
    if len(conflicts):
        log.warning(f"{len(conflicts)} structures have conflicting labels across duplicate entries")

    df_sorted = df.sort_values(label_col, ascending=False)  # 1 (concern) before 0
    df_dedup = df_sorted.drop_duplicates(subset="smiles_std", keep="first").reset_index(drop=True)
    log.info(f"Deduplication: {before} -> {len(df_dedup)} entries")
    return df_dedup


# ---------------------------------------------------------------------------
# Step 4: Binary label
# ---------------------------------------------------------------------------
def build_binary_label(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.copy()
    normalized = df[label_col].astype(str).str.strip().str.lower()
    unmapped = set(normalized) - set(LABEL_MAP.keys())
    if unmapped:
        log.warning(f"Unrecognized category strings after normalization: {unmapped}")

    df["dili_label"] = normalized.map(LABEL_MAP)
    n_ambiguous_or_unmapped = df["dili_label"].isna().sum()
    df = df[df["dili_label"].notna()].reset_index(drop=True)
    df["dili_label"] = df["dili_label"].astype(int)
    log.info(f"Dropped {n_ambiguous_or_unmapped} Ambiguous / unmapped entries")
    log.info(f"Final class balance:\n{df['dili_label'].value_counts()}")
    return df


# ---------------------------------------------------------------------------
# Step 5: Sanity-check plot
# ---------------------------------------------------------------------------
def plot_class_distribution(df: pd.DataFrame, out_path: Path):
    if df.empty:
        log.warning("No processed rows to plot — skipping class distribution plot.")
        return
    counts = df["dili_label"].value_counts().reindex([0, 1], fill_value=0)
    labels = ["No DILI Concern", "DILI Concern"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center")
    ax.set_ylabel("Number of drugs")
    ax.set_title("DrugSafe — Class distribution (post-cleaning)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info(f"Saved class distribution plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_raw_dilirank()
    df = resolve_smiles(df, name_col=COLUMN_MAP["drug_name_col"])
    df = standardize_dataframe(df)
    df = deduplicate(df, label_col=COLUMN_MAP["label_col"])
    df = build_binary_label(df, label_col=COLUMN_MAP["label_col"])

    out_cols = [COLUMN_MAP["drug_name_col"], "smiles_std", COLUMN_MAP["label_col"], "dili_label"]
    df[out_cols].to_csv(PROCESSED_PATH, index=False)
    log.info(f"Saved {len(df)} processed entries to {PROCESSED_PATH}")

    plot_class_distribution(df, FIGURES_DIR / "class_distribution.png")


if __name__ == "__main__":
    main()
