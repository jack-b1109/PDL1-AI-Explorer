# PDL1 AI Explorer

### An IGNITE-Inspired Replication of AI-Powered Tumour Cell Detection

**Exploring the intersection of deep learning, computational pathology and cancer immunotherapy.**

---

## Overview

This project is a replication of research into the use of deep learning for detecting PD-L1 expression in non-small cell lung cancer (NSCLC) tissue samples.

Building on the IGNITE research project, I explored how a YOLOv5-based computer vision model can identify and classify different cell types in digitised histopathology images.

The project focuses on three classes:

- **PD-L1-negative tumour cells**
- **PD-L1-positive tumour cells**
- **Non-tumour cells**

The aim was to reproduce the model's inference workflow, examine its predictions against human annotations and make the results accessible through an interactive web application.

## Project Results

| Metric                   | Result |
| ------------------------ | -----: |
| Test images processed    |     98 |
| AI detections generated  | 17,330 |
| Visualisations generated |     98 |
| Human readers compared   |      6 |

### Model-versus-human F1 scores

The following scores represent agreement between model predictions and individual human readers.

| Reader | Group | F1 Score |
| ------ | ----- | -------: |
| P1     | RUMC  |     0.54 |
| P2     | RUMC  |     0.70 |
| P3     | RUMC  |     0.61 |
| P4     | SCDC  |     0.58 |
| P5     | SCDC  |     0.60 |
| P6     | SCDC  |     0.61 |

These scores measure agreement with human annotations, not clinical accuracy or the ability to predict treatment response.

## Interactive Explorer

The project includes a Streamlit application that allows users to:

- Explore AI-generated histopathology visualisations.
- Examine predicted cell classifications.
- Investigate detection statistics.
- Compare model predictions with human annotations.
- Understand the research methodology and its potential relevance to immunotherapy.

**Live application:** [Open the PD-L1 AI Explorer](YOUR_STREAMLIT_URL)

## Methodology

The replication workflow involved:

1. Exploring the original IGNITE research, dataset and available pretrained models.
2. Setting up the PyTorch environment and GPU-based inference workflow.
3. Processing the 98-image PD-L1 test set.
4. Generating AI predictions and visualisations.
5. Comparing model outputs against human annotations using F1 scores.
6. Developing and deploying an interactive Streamlit application to communicate the findings.

## Technology Stack

- Python
- PyTorch
- YOLOv5
- NumPy
- Matplotlib
- Streamlit
- Google Colab
- GitHub

## Research Background

This project builds on the following research:

Spronck, J., van Eekelen, L., et al. (2025). *A tissue and cell-level annotated H&E and PD-L1 histopathology image dataset in non-small cell lung cancer.*

- [Research paper – arXiv](https://arxiv.org/abs/2507.16855)
- [IGNITE Data Toolkit – GitHub](https://github.com/DIAGNijmegen/ignite-data-toolkit)
- [Dataset – Zenodo](https://zenodo.org/records/17735903)

## Limitations

This project is an educational research replication, not a clinically validated diagnostic system.

Model predictions are evaluated against human annotations and should not be interpreted as definitive diagnoses, measures of treatment effectiveness or recommendations for patient care.

## Acknowledgements

I would like to acknowledge the researchers behind the original IGNITE project for making their research, annotated dataset and tools available for further exploration.

This project was completed as part of TKS, with a focus on understanding and reproducing existing research through practical experimentation.

---

*Exploring how artificial intelligence could help us see cancer pathology through a different lens.*
