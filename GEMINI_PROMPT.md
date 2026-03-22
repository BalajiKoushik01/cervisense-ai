# Colab Gemini Instructions

Hello Gemini! The user has uploaded this project to Google Colab. Your exact objective is to orchestrate the CerviSense-AI training pipeline using the provided files.

## What to do step-by-step:
1. **Unzip & Clean Up**: First, execute a command to unzip `cervisense-ai_export.zip` into the `/content/cervisense-ai` directory if it hasn't been unzipped. Once unzipped, you MUST safely permanently delete `cervisense-ai_export.zip` to free up disk space on the Colab environment.
2. **Analyze the Project**: Quickly review `README.md` and `requirements_colab.txt` to understand the training parameters and necessary PIP installations (e.g. `pip install -r requirements_colab.txt`).
3. **Execute the Notebooks**: 
   - Open and execute `notebooks/02_ssl_training.ipynb` sequentially.
   - Or, run the individual python scripts sequentially via terminal (`python training/train_ssl.py --config configs/ssl_config.yaml`).
4. **Hardware Verification**: Ensure that the Colab GPU (T4/A100) is being utilized by running `nvidia-smi` and checking PyTorch CUDA availability.

Note: Do not keep unnecessary cached items. Your goal is to maximize the free disk space and begin GPU training immediately!
