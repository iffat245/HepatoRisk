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
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# The 2D structure-drawing part of RDKit depends on system graphics
# libraries (libXrender, etc.) that aren't always available on every
# hosting platform. Rather than let a missing system library crash the
# entire app, we make the drawing feature optional: if it's available, we
# use it; if not, the app still works fully, just without the picture.
try:
    from rdkit.Chem import Draw
    DRAWING_AVAILABLE = True
except ImportError:
    DRAWING_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config — paths are relative to wherever you run `streamlit run` from.
# Run this from the project root (drugsafe\), not from inside app\.
# ---------------------------------------------------------------------------
FP_RADIUS = 2
FP_NBITS = 2048
DESCRIPTOR_NAMES = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "RingCount", "AromaticRings"]

MODEL_PATH = Path("models/final_model.joblib")
PROCESSED_PATH = Path("data/processed/dilirank_processed.csv")

st.set_page_config(page_title="DrugSafe", page_icon="🧪", layout="centered")

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

# A small built-in fallback for common drugs, checked BEFORE hitting
# PubChem. These always work instantly, even if PubChem is down, rate-
# limiting you, or changes its response format again — useful for demos
# and for the sanity-check tests you'll run most often.
KNOWN_DRUG_SMILES = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "isoniazid": "NNC(=O)c1ccncc1",
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "amoxicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "diazepam": "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1",
    "omeprazole": "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C",
}


@st.cache_data(show_spinner=False)
def lookup_smiles_by_name(drug_name: str) -> str | None:
    """
    Look up a drug name and return its SMILES, or None if not found.
    Checks the built-in fallback list first (instant, no internet needed),
    then falls back to a live PubChem lookup for anything not in that list.
    """
    normalized = drug_name.strip().lower()
    if normalized in KNOWN_DRUG_SMILES:
        return KNOWN_DRUG_SMILES[normalized]

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
st.title("🧪 DrugSafe")
st.caption("An explainable in-silico DILI-concern classifier")

st.warning(
    "**Research and educational tool only.** Trained on a public reference dataset "
    "(FDA DILIrank 2.0). This does **not** provide medical advice, does **not** predict "
    "individual patient safety, and must **not** be used for clinical or regulatory "
    "decisions. It estimates statistical association between molecular structure and a "
    "DILI-concern category — it does not determine whether any drug is clinically 'safe' "
    "or 'unsafe.'"
)

input_mode = st.radio("Search by", ["SMILES", "Drug name"], horizontal=True)

if input_mode == "SMILES":
    smiles_input = st.text_input(
        "Enter a SMILES string",
        placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O  (this example is aspirin)",
    )
else:
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
                "combination products, or unusual name spellings. Try the SMILES option "
                "instead if you have the structure, or double-check the spelling."
            )
        else:
            st.caption(f"Found on PubChem: `{looked_up_smiles}`")
            smiles_input = looked_up_smiles

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
            if DRAWING_AVAILABLE:
                st.image(Draw.MolToImage(mol, size=(280, 280)))
            else:
                st.info("2D structure image isn't available on this deployment, but the prediction below is unaffected.")

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
    "DrugSafe is a portfolio/research project built on the FDA DILIrank 2.0 dataset. "
    "It is not a certified diagnostic or regulatory tool, has not been clinically validated, "
    "and should never be used to make decisions about any individual's treatment."
)
