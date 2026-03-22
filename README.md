# CerviSense-AI: Self-Supervised Cross-Modal Fusion for Cervical Cancer Diagnostics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

CerviSense-AI is a state-of-the-art, dual-domain cervical cancer diagnostic engine. It utilizes **Self-Supervised Learning (MoCo-v3)** on an `EfficientNetV2-S` backbone to learn robust representations from unlabelled clinical data, followed by **Hierarchical Cross-Modal Attention Fusion (HCMAF)** to intelligently combine cytology and colposcopy modalities. 

This repository provides an end-to-end industrial-grade pipeline designed for reproducibility, scalability, and Explainable AI (XAI) transparency.

---

## 📑 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Datasets](#datasets)
3. [Environment Setup](#environment-setup)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation & XAI](#evaluation--xai)
6. [Industrial Compliance](#industrial-compliance)
7. [Contributing](#contributing)
8. [License & Citation](#license--citation)

---

## 🏗 Architecture Overview
The pipeline consists of three core phases:

1. **Self-Supervised Pre-Training (SSL):** Momentum Contrast (MoCo-v3) is used to pre-train the EfficientNetV2-S encoders on a large corpus of unlabelled cytology and histology images, learning rich feature representations without manual annotations.
2. **Supervised Fine-Tuning:** The pre-trained encoders are fine-tuned using Focal Loss to combat the severe class imbalance inherent to cervical cancer datasets (e.g., disproportionate "Normal" vs "CIN3" distributions).
3. **Cross-Modal Fusion (HCMAF):** For multi-modal inputs, a Hierarchical Cross-Modal Attention mechanism fuses cytology and colposcopy features. A gating mechanism determines the optimal reliance on each modality dynamically.

**Explainable AI (XAI):** Built-in support for Grad-CAM++, t-SNE, UMAP, and ROC analysis guarantees clinician trust.

## 📊 Datasets
This framework is designed to train on industry-standard benchmarking datasets:
- **SIPaKMeD** (4,049 cytology images)
- **CRIC Cervix** (11,534 cropped cells)
- **AnnoCerv** (532+ colposcopy images)
- **Intel MobileODT** (1,500+ cervical photographs)

## ⚙️ Environment Setup

### Local Installation
```bash
# Clone the repository
git clone https://github.com/BalajiKoushik01/cervisense-ai.git
cd cervisense-ai

# Create a virtual environment
python -m venv cervisense_env

# Activate (Windows)
.\cervisense_env\Scripts\activate
# Activate (Linux/Mac)
source cervisense_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Training
A bundled export system is available for Google Colab targeting T4 GPUs.
1. Run `python scripts/export_to_colab.py` locally to generate `cervisense-ai_export.zip`.
2. Upload the zip to the root of your Google Drive (`MyDrive/`).
3. Upload `notebooks/02_ssl_training.ipynb` to Colab and execute the required cells.

## 🚀 Training Pipeline

### Phase 0: Data Preparation
Ensure datasets are placed in `data/raw/`. 
```bash
python data_utils/preprocess.py
```
This script handles image validation, perceptual hashing (deduplication), and split generation.

### Phase 1: SSL Pre-Training
```bash
python training/train_ssl.py --config configs/ssl_config.yaml
```

### Phase 2: Supervised Fine-Tuning
```bash
python training/train_finetune.py --config configs/finetune_config.yaml
```

### Phase 3: Cross-Modal Fusion
```bash
python training/train_fusion.py --config configs/fusion_config.yaml
```

## 📈 Evaluation & XAI
Evaluate your absolute best checkpoint and generate publication-ready heatmaps:
```bash
python evaluation/evaluate.py --checkpoint checkpoints/finetune/best_finetune.pth
python visualization/viz_runner.py --checkpoint checkpoints/finetune/best_finetune.pth
```
Output visualizations are stored in `outputs/plots/`.

## 🏭 Industrial Compliance
- **Code Quality:** Fully compliant with SonarQube's strict Cognitive Complexity requirements (`CC < 15`).
- **Mixed Precision:** Uses `torch.amp.autocast` for scalable performance on NVIDIA hardware.
- **Linting:** Enforced standardized formatting via Pyre & PyLint.

## 🤝 Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, syntax guidelines, and the process for submitting Pull Requests.

## 📄 License & Citation
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

If you utilize CerviSense-AI in your research, please cite our repository via `CITATION.cff`.
