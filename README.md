# CerviSense-AI — cervical-image machine-learning research

CerviSense-AI contains research code for training and evaluating image representations. It includes self-supervised learning components, classifier fine-tuning, datasets, preprocessing, metrics and visualisation scripts.

**Research use only.** This code is **not a medical device**, is not clinically validated in this repository and must not be used to diagnose, screen or triage patients. No clinical sensitivity, specificity or regulatory approval is claimed.

## Code organisation

- `models/` — encoder, classification, contrastive learning and fusion components.
- `training/` — self-supervised and supervised training scripts.
- `data_utils/` — dataset loading, preprocessing and augmentation.
- `evaluation/` — metric calculation and evaluation entry point.
- `configs/` — training configuration files.
- `notebooks/` — exploratory and Colab-oriented notebooks.
- `scripts/` — dataset and export helpers.

## Setup

A Python environment with the dependencies in `requirements.txt` is needed. GPU support depends on the installed PyTorch build and local drivers.

```bash
git clone https://github.com/BalajiKoushik01/cervisense-ai.git
cd cervisense-ai
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Datasets are not bundled with the repository. Obtain each dataset from its owner under the applicable licence and usage conditions; follow the expected directory structure in the dataset and training code.

## Training and evaluation

```bash
python training/train_ssl.py --config configs/ssl_config.yaml
python evaluation/evaluate.py --checkpoint PATH_TO_CHECKPOINT
```

The evaluation script expects a compatible checkpoint and test data (default path: `data/combined/supervised/test`). Checkpoints are not guaranteed to be available. Examine the model's expected classes and the evaluation protocol before interpreting metrics.

For Colab packaging, `scripts/export_to_colab.py` exists; review its hard-coded relative directories before running it from another working directory.

## Interpreting results

Do not report accuracy or clinical benefit without a reproducible dataset definition, patient-level separation, external validation, confidence intervals and documented failure cases. Visual explanations such as saliency maps are not clinical evidence. Any use involving patient information also requires privacy, ethics and regulatory review.

The repository's `CITATION.cff` and `CONTRIBUTING.md` provide attribution and contribution guidance.
