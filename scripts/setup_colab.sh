#!/bin/bash
# Mount Google Drive, Clone Repo, Setup Env
echo "Setting up Colab environment..."

# Setup wandb token if secret provides
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY=$WANDB_API_KEY
    echo "Wandb Key Configured"
fi

# Install requirements
pip install -r requirements_colab.txt -q

# Symlink data to Drive (assuming Drive is mounted at /content/drive via notebook)
mkdir -p /content/drive/MyDrive/cervisense/data
rm -rf /content/cervisense-ai/data
ln -s /content/drive/MyDrive/cervisense/data /content/cervisense-ai/data

# Verify GPU
nvidia-smi

echo "Setup script completed."
