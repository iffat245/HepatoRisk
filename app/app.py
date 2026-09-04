"""
DrugSafe — Day 5: Streamlit Application
==========================================
Run:    streamlit run app/app.py

Requires (in addition to earlier days' packages):
    pip install streamlit
"""

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib
import shap

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Config — paths are relative to wherever you run `streamlit run` from.
# Run this from the project root (drugsafe\), not from inside app\.
# ---------------------------------------------------------------------------
FP_RADIUS = 2
FP_NBITS = 2048
DESCRIPTOR_NAMES = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "RingCount", "AromaticRings"]

MODEL_PATH = Path("models/final_model.joblib")
PROCESSED_PATH = Path("data/processed/dilirank_processed.csv")

st.set_page_config(page_title="HepatoRisk", page_icon="🧪", layout="centered")

_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


# ---------------------------------------------------------------------------
# Cached loaders — @st.cache_resource means these only run ONCE, not on
# every interaction, so the app stays fast.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_reference_fingerprints():
    """Training-set fingerprints, used for the similarity/applicability check."""
    df = pd.read_csv(PROCESSED_PATH)
    fps = []
    for smi in df["smiles_std"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS))
    return fps


# ---------------------------------------------------------------------------
# Chemistry helpers (same logic as Day 1/2/4 scripts, kept self-contained
# here so the app doesn't depend on importing other script files)
# ---------------------------------------------------------------------------
def standardize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        mol = _largest_fragment.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None, None
    return mol, Chem.MolToSmiles(mol, canonical=True)


PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
    "CanonicalSMILES/JSON"
)

@st.cache_resource
def load_builtin_library(n_samples: int = 20) -> dict[str, str]:
    """
    Library for the 'Choose from library' option. Preferentially built from
    YOUR OWN processed dataset (dilirank_processed.csv) — every smiles_std
    value there already passed through RDKit standardization in
    data_prep.py, so it's guaranteed to parse and reflects real compounds
    from your actual pipeline, not a hardcoded external guess.

    Falls back to a small set of well-known, RDKit-verified drugs only if
    that file isn't available yet, so this option still always works.
    """
    if PROCESSED_PATH.exists():
        df = pd.read_csv(PROCESSED_PATH)
        if {"CompoundName", "smiles_std"}.issubset(df.columns) and len(df) > 0:
            df = df.dropna(subset=["CompoundName", "smiles_std"]).drop_duplicates(subset="CompoundName")
            sample = df.sample(n=min(n_samples, len(df)), random_state=42).sort_values("CompoundName")
            return dict(zip(sample["CompoundName"], sample["smiles_std"]))

    return dict(_FALLBACK_LIBRARY)


# Fallback only — used if dilirank_processed.csv isn't found. Every entry
# below was verified to parse correctly with RDKit before being included.
_FALLBACK_LIBRARY = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Acetaminophen (Paracetamol)": "CC(=O)Nc1ccc(O)cc1",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "Diazepam": "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1",
    "Omeprazole": "CC1=CN=C(C(=C1OC)C)CS(=O)C1=NC2=C(N1)C=C(C=C2)OC",
    "Penicillin G": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "Amoxicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "Codeine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](OC)C=C[C@H]3[C@H]1C5",
    "Nicotine": "CN1CCC[C@H]1c1cccnc1",
    "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "Simvastatin": "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12",
    "Diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Metronidazole": "Cc1ncc([N+](=O)[O-])n1CCO",
    "Ciprofloxacin": "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Fluoxetine": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "Losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(cc1)-c1ccccc1-c1nnn[nH]1",
    "Amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl",
}


@st.cache_data(show_spinner=False)
def lookup_smiles_by_name(drug_name: str) -> str | None:
    """
    Look up a drug name on PubChem and return its SMILES, or None if not
    found. Cached so re-searching the same name during your session is
    instant and doesn't hit PubChem again.
    """
    encoded_name = quote(drug_name.strip(), safe="")
    url = PUBCHEM_URL.format(name=encoded_name)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        props = resp.json()["PropertyTable"]["Properties"][0]
        # PubChem has been transitioning the property name from
        # "CanonicalSMILES" to "ConnectivitySMILES" — check both so this
        # keeps working regardless of which one a given response uses.
        return props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or props.get("IsomericSMILES")
    except Exception:
        return None


def compute_features(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
    fp_arr = np.zeros((FP_NBITS,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)

    descriptors = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }
    X = np.concatenate([fp_arr, list(descriptors.values())]).reshape(1, -1)
    return X, fp, descriptors


def similarity_tier(query_fp, reference_fps, k=5):
    sims = sorted(DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps), reverse=True)
    max_sim = sims[0]
    mean_top_k = sum(sims[:k]) / k
    if mean_top_k >= 0.5:
        tier = "High"
    elif mean_top_k >= 0.3:
        tier = "Medium"
    else:
        tier = "Low"
    return max_sim, mean_top_k, tier


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.title("🧪 HepatoRisk")
st.caption("An explainable in-silico DILI-concern classifier")

st.warning(
    "**Research and educational tool only.** Trained on a public reference dataset "
    "(FDA DILIrank 2.0). This does **not** provide medical advice, does **not** predict "
    "individual patient safety, and must **not** be used for clinical or regulatory "
    "decisions. It estimates statistical association between molecular structure and a "
    "DILI-concern category — it does not determine whether any drug is clinically 'safe' "
    "or 'unsafe.'"
)

input_mode = st.radio("Search by", ["SMILES", "Drug name", "Choose from library"], horizontal=True)

if input_mode == "SMILES":
    smiles_input = st.text_input(
        "Enter a SMILES string",
        placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O  (this example is aspirin)",
    )
elif input_mode == "Drug name":
    drug_name_input = st.text_input(
        "Enter a drug name",
        placeholder="e.g. Aspirin",
    )
    smiles_input = None
    if drug_name_input:
        with st.spinner(f"Looking up '{drug_name_input}' on PubChem..."):
            looked_up_smiles = lookup_smiles_by_name(drug_name_input)
        if looked_up_smiles is None:
            st.error(
                f"Couldn't find '{drug_name_input}' on PubChem. This can happen for "
                "biologics (antibodies, insulins, etc. — these don't have a SMILES code), "
                "combination products, unusual name spellings, or no internet connection. "
                "Try the 'Choose from library' option for a guaranteed-working example, or "
                "the SMILES option if you have the structure."
            )
        else:
            st.caption(f"Found on PubChem: `{looked_up_smiles}`")
            smiles_input = looked_up_smiles
else:  # "Choose from library"
    st.caption(
        "A small built-in set of well-known drugs — no internet or PubChem lookup needed, "
        "so this option always works."
    )
    library_choice = st.selectbox(
        "Pick a drug",
        options=list(load_builtin_library().keys()),
    )
    smiles_input = load_builtin_library()[library_choice]
    st.caption(f"SMILES: `{smiles_input}`")

if smiles_input:
    mol, std_smiles = standardize_smiles(smiles_input)

    if mol is None:
        st.error("This doesn't parse as a valid molecule. Double-check the SMILES string.")
    else:
        st.success(f"Valid molecule. Standardized SMILES: `{std_smiles}`")

        X, query_fp, descriptors = compute_features(mol)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Molecular properties")
            st.table(pd.DataFrame(descriptors.items(), columns=["Property", "Value"]))
        with col2:
            st.subheader("Structure")
            st.image(Draw.MolToImage(mol, size=(280, 280)))

        # --- Prediction ---
        model = load_model()
        prob = model.predict_proba(X)[0, 1]
        prediction = "DILI Concern" if prob >= 0.5 else "No DILI Concern"

        st.subheader("Prediction")
        c1, c2 = st.columns(2)
        c1.metric("DILI-concern probability", f"{prob:.1%}")
        c2.metric("Classification", prediction)

        # --- Explanation ---
        st.subheader("Why did the model predict this?")
        with st.spinner("Computing explanation..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X, check_additivity=False)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            elif np.asarray(shap_values).ndim == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

        feature_names = [f"fp_bit_{i}" for i in range(FP_NBITS)] + DESCRIPTOR_NAMES
        top_idx = np.argsort(-np.abs(sv))[:8]
        exp_df = pd.DataFrame({
            "feature": [feature_names[i] for i in top_idx],
            "impact on prediction": [sv[i] for i in top_idx],
        }).set_index("feature")
        st.bar_chart(exp_df)
        st.caption(
            "Positive bars push the prediction toward 'DILI Concern'; negative bars push it "
            "toward 'No DILI Concern.' Bars labeled 'fp_bit_N' represent specific chemical "
            "substructures; the other labels are the plain molecular properties above."
        )

        # --- Applicability domain ---
        st.subheader("Chemical similarity to training data")
        reference_fps = load_reference_fingerprints()
        max_sim, mean_sim, tier = similarity_tier(query_fp, reference_fps)

        tier_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[tier]
        st.write(f"{tier_color} **{tier}** similarity (closest training match: {max_sim:.2f}, "
                 f"average of 5 nearest: {mean_sim:.2f})")

        if tier == "Low":
            st.warning(
                "This molecule is chemically dissimilar to anything in the training data. "
                "The prediction above may be unreliable outside the model's chemical domain — "
                "treat it with reduced confidence."
            )
        st.caption(
            "This is a similarity heuristic based on Tanimoto similarity of Morgan fingerprints, "
            "not a formally validated applicability domain."
        )

st.divider()
st.caption(
    "HepatoRisk is a portfolio/research project built on the FDA DILIrank 2.0 dataset. "
    "It is not a certified diagnostic or regulatory tool, has not been clinically validated, "
    "and should never be used to make decisions about any individual's treatment."
)
