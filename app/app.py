"""
HepatoRisk — Day 5: Streamlit Application
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

# Importing rdkit.Chem.Draw (in any form, including rdMolDraw2D) fails on
# this deployment because the subpackage's own setup code depends on a
# system graphics library (libXrender) that isn't available here. Wrapping
# the import means the app runs fully — prediction, explanation, similarity
# check — just without the 2D structure picture, instead of crashing.
try:
    from rdkit.Chem.Draw import rdMolDraw2D
    DRAWING_AVAILABLE = True
except ImportError:
    DRAWING_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FP_RADIUS = 2
FP_NBITS = 2048
DESCRIPTOR_NAMES = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotatableBonds", "RingCount", "AromaticRings"]

MODEL_PATH = Path("models/final_model.joblib")
PROCESSED_PATH = Path("data/processed/dilirank_processed.csv")

st.set_page_config(page_title="HepatoRisk", page_icon="🧪", layout="centered")

_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()

# A built-in library of common drugs, checked BEFORE hitting PubChem.
# These always work instantly, even if PubChem is down, rate-limiting you,
# or changes its response format again — and they power the "Choose from
# library" option below, which never depends on the internet at all.
KNOWN_DRUG_SMILES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Paracetamol (Acetaminophen)": "CC(=O)Nc1ccc(O)cc1",
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Isoniazid": "NNC(=O)c1ccncc1",
    "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "Amoxicillin": "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "Diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "Warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "Diazepam": "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1",
    "Omeprazole": "COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C",
    "Simvastatin": "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12",
    "Metronidazole": "Cc1ncc([N+](=O)[O-])n1CCO",
    "Ciprofloxacin": "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Fluoxetine": "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "Losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
    "Amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl",
    "Naproxen": "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "Cetirizine": "OC(=O)COCCN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1",
    "Ranitidine": "CNC(=C[N+](=O)[O-])NCCSCc1ccc(o1)CN(C)C",
}

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
    "CanonicalSMILES/JSON"
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_reference_fingerprints():
    df = pd.read_csv(PROCESSED_PATH)
    fps = []
    for smi in df["smiles_std"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS))
    return fps


@st.cache_data(show_spinner=False)
def lookup_smiles_by_name(drug_name: str) -> str | None:
    """
    Look up a drug name and return its SMILES, or None if not found.
    Checks the built-in library first (instant, no internet needed), then
    falls back to a live PubChem lookup for anything not in that list.
    """
    normalized = drug_name.strip().lower()
    for known_name, smi in KNOWN_DRUG_SMILES.items():
        if known_name.lower().startswith(normalized) or normalized in known_name.lower():
            return smi

    encoded_name = quote(drug_name.strip(), safe="")
    url = PUBCHEM_URL.format(name=encoded_name)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        props = resp.json()["PropertyTable"]["Properties"][0]
        # PubChem has been transitioning the property name from
        # "CanonicalSMILES" to "ConnectivitySMILES" — check both.
        return props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or props.get("IsomericSMILES")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chemistry helpers
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


def mol_to_svg(mol, size=(320, 320)) -> str | None:
    """
    Draws the molecule as an SVG. Returns None if RDKit's drawing module
    isn't available on this deployment (see the import guard above).
    """
    if not DRAWING_AVAILABLE:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


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

smiles_input = None

if input_mode == "SMILES":
    smiles_input = st.text_input(
        "Enter a SMILES string",
        placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O  (this example is aspirin)",
    )

elif input_mode == "Drug name":
    drug_name_input = st.text_input("Enter a drug name", placeholder="e.g. Aspirin")
    if drug_name_input:
        with st.spinner(f"Looking up '{drug_name_input}'..."):
            looked_up_smiles = lookup_smiles_by_name(drug_name_input)
        if looked_up_smiles is None:
            st.error(
                f"Couldn't find '{drug_name_input}'. This can happen for biologics "
                "(antibodies, insulins, etc. — these don't have a SMILES code), combination "
                "products, or unusual name spellings. Try the SMILES option instead if you "
                "have the structure, or try the library option for a guaranteed-working example."
            )
        else:
            st.caption(f"Found: `{looked_up_smiles}`")
            smiles_input = looked_up_smiles

else:  # Choose from library
    chosen_drug = st.selectbox("Pick a drug", list(KNOWN_DRUG_SMILES.keys()))
    smiles_input = KNOWN_DRUG_SMILES[chosen_drug]
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
            svg = mol_to_svg(mol)
            if svg is not None:
                st.image(svg, use_container_width=True)
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
    "HepatoRisk is a portfolio/research project built on the FDA DILIrank 2.0 dataset. "
    "It is not a certified diagnostic or regulatory tool, has not been clinically validated, "
    "and should never be used to make decisions about any individual's treatment."
)