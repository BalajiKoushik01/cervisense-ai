# CerviSense-AI

> Self-supervised cross-modal fusion framework for cervical cancer diagnostics.
> Dual-domain SSL (colposcopy + histopathology) → 5-class CIN grading → XAI heatmaps.

## Architecture
- Backbone: EfficientNetV2-S (21M params)
- SSL: MoCo-v3 (momentum encoder + cosine LR)
- Fusion: HCMAF (Hierarchical Cross-Modal Attention Fusion)
- XAI: Grad-CAM++ + SHAP + t-SNE/UMAP

## Quick Start
1. Download datasets (see Section 3 of INSTRUCTIONS.md)
2. Run: python data_utils/preprocess.py
3. Train SSL: python training/train_ssl.py --config configs/ssl_config.yaml
4. Fine-tune: python training/train_finetune.py --config configs/finetune_config.yaml
5. Evaluate: python evaluation/evaluate.py --checkpoint checkpoints/finetune/best_finetune.pth

## Results
[To be filled after training]

## Datasets Used
- SIPaKMeD (cytology, 4,049 images)
- CRIC Cervix (cytology, 11,534 cells)
- AnnoCerv (colposcopy, 532+ images)
- Intel MobileODT (cervical photography, 1,500+ images)
- Herlev (PAP smear, 917 images)
