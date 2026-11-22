
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile

st.set_page_config(
    page_title="PD-L1 Explorer",
    page_icon="🔬",
    layout="wide"
)


ROOT = Path(__file__).parent

# Automatically restore the project data from the two uploaded archives
for archive_name in ["PD-L1-Data-Part-1.zip", "PD-L1-Data-Part-2.zip"]:
    archive_path = ROOT / archive_name
    if archive_path.exists():
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(ROOT)

DATA = ROOT / "data"
VISUALS = DATA / "figures/pdl1/pdl1"
PREDICTIONS = DATA / "inference/pdl1/pdl1"

visual_files = sorted([
    f for f in VISUALS.glob("*.jpg")
    if (PREDICTIONS / (f.stem + ".npy")).exists()
])

prediction_files = sorted(PREDICTIONS.glob("*.npy"))

@st.cache_data
def get_total_detections(paths):
    return sum(len(np.load(p)) for p in paths)

total = get_total_detections([str(p) for p in prediction_files])

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
.stApp { background:#F7F7F2; color:#19352F; }
.block-container {max-width:1200px;padding-top:2rem;padding-bottom:4rem;}
h1,h2,h3 {font-family:'Playfair Display',serif!important;color:#19352F!important;}
.hero {padding:55px 0 40px;border-bottom:1px solid #E3E6DF;}
.eyebrow {font-size:11px;letter-spacing:2.5px;font-weight:700;color:#527565;}
.hero-title {font-family:'Playfair Display',serif;font-size:clamp(42px,6vw,66px);line-height:1.1;color:#19352F;}
.hero-desc {color:#718078;font-size:16px;line-height:1.8;max-width:700px;margin-top:18px;}
.metric {background:white;border:1px solid #E3E6DF;border-radius:12px;padding:22px;}
.metric-value {font-family:'Playfair Display',serif;font-size:34px;color:#19352F;}
.metric-label {font-size:11px;letter-spacing:1.5px;color:#718078;margin-top:5px;}
.section-label {font-size:11px;letter-spacing:2px;font-weight:700;color:#527565;}
.notice {background:#EAF0E9;border-left:3px solid #668875;padding:15px 18px;border-radius:4px;color:#365649;font-size:13px;line-height:1.7;}
div[data-testid="stImage"] img {border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="eyebrow">PD-L1 / EXPLORER</div>
<div class="hero">
<div class="section-label">COMPUTATIONAL PATHOLOGY · RESEARCH</div>
<div class="hero-title">Seeing cancer<br>through a different lens.</div>
<div class="hero-desc">
Explore AI-generated cell detections in non-small cell lung cancer
tissue, and examine how model predictions compare with human expert
annotations.
</div>
</div>
""", unsafe_allow_html=True)

st.write("")
a,b,c = st.columns(3)
for col,value,label in [
    (a, "98", "TEST TISSUE IMAGES"),
    (b, f"{total:,}", "AI CELL DETECTIONS"),
    (c, "6", "HUMAN READERS")
]:
    col.markdown(
        f'<div class="metric"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True
    )

st.write("")
st.markdown(
    '<div class="notice"><b>Research demonstration only.</b> '
    'This application presents outputs from a pretrained research model. '
    'It is not a clinical diagnostic tool.</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

st.markdown('<div class="section-label">01 / VISUAL EXPLORATION</div>',
            unsafe_allow_html=True)
st.header("Inside the tissue")
st.write("Select a sample to explore its generated research visualisation.")

if visual_files:
    selected = st.selectbox(
        "Choose a tissue sample",
        visual_files,
        format_func=lambda p: p.name
    )
    st.image(str(selected), use_container_width=True)

    pred_file = PREDICTIONS / (selected.stem + ".npy")
    detections = np.load(pred_file)

    st.metric("AI detections in selected sample", f"{len(detections):,}")

    if len(detections):
        counts = pd.Series(
            detections[:,2].astype(int)
        ).value_counts().reindex([0,1,2],fill_value=0)

        st.markdown("#### Predicted cell categories")
        st.bar_chart(pd.DataFrame({
            "Detected cells": counts.values
        }, index=[
            "PD-L1-negative tumour",
            "PD-L1-positive tumour",
            "Non-tumour"
        ]))
else:
    st.warning("No matching visualisations and predictions were found.")

st.divider()
st.markdown('<div class="section-label">02 / MODEL EVALUATION</div>',
            unsafe_allow_html=True)
st.header("AI meets human expertise")
st.write(
    "Macro F1 scores comparing AI predictions with individual "
    "human readers using the original research evaluation."
)

scores = pd.DataFrame({
    "Human reader": ["P1","P2","P3","P4","P5","P6"],
    "Research group": ["RUMC","RUMC","RUMC","SCDC","SCDC","SCDC"],
    "Macro F1": [0.54,0.70,0.61,0.58,0.60,0.61]
})

left,right = st.columns(2)
with left:
    st.markdown("**RUMC readers**")
    st.bar_chart(scores[scores["Research group"]=="RUMC"].set_index("Human reader")[["Macro F1"]])
with right:
    st.markdown("**SCDC readers**")
    st.bar_chart(scores[scores["Research group"]=="SCDC"].set_index("Human reader")[["Macro F1"]])

st.dataframe(scores, hide_index=True, use_container_width=True)

st.markdown(
    '<div class="notice">These F1 scores represent agreement with '
    'individual human readers. They do not establish clinical accuracy '
    'or treatment effectiveness.</div>',
    unsafe_allow_html=True
)

st.divider()
st.markdown('<div class="section-label">03 / THE SCIENCE</div>',
            unsafe_allow_html=True)
st.header("Why PD-L1 matters")

x,y,z = st.columns(3)
with x:
    st.subheader("The biomarker")
    st.write("PD-L1 is a protein involved in regulating immune responses. Its expression can be relevant to cancer immunotherapy.")
with y:
    st.subheader("The technology")
    st.write("A pretrained YOLOv5 model identifies PD-L1-positive tumour cells, PD-L1-negative tumour cells and non-tumour cells.")
with z:
    st.subheader("The evaluation")
    st.write("AI predictions are compared with expert annotations to investigate agreement in cell identification.")

with st.expander("Dataset and research methodology"):
    st.write("""
    This project uses the PD-L1 test-set outputs from the IGNITE
    data toolkit research project. The original workflow uses a
    pretrained model to identify three cell categories and evaluates
    its predictions against human-reader annotations.

    The original evaluation uses a 20-pixel spatial matching radius.
    """)

with st.expander("Research acknowledgement"):
    st.write("""
    Spronck, J., van Eekelen, L., et al. (2025).
    A tissue and cell-level annotated H&E and PD-L1 histopathology
    image dataset in non-small cell lung cancer.

    Dataset: https://zenodo.org/records/17735903

    Code: https://github.com/DIAGNijmegen/ignite-data-toolkit
    """)

st.markdown("---")
st.caption("PD-L1 / EXPLORER · TKS ITERATE · 2026")
