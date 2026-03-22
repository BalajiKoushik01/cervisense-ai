"""
setup_kaggle_token.py  – Run ONCE to store your Kaggle API token.
Usage: python scripts/setup_kaggle_token.py
"""
import os
import json
import sys

print("=" * 60)
print("Kaggle API Token Setup")
print("=" * 60)
print()
print("Step 1: Go to https://www.kaggle.com/settings")
print("Step 2: Scroll to 'API' section")
print("Step 3: Click 'Create New Token'  (saves kaggle.json to Downloads)")
print()

username = input("Enter your Kaggle username: ").strip()
api_key  = input("Enter your Kaggle API key : ").strip()

kaggle_dir  = os.path.join(os.path.expanduser("~"), ".kaggle")
kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

os.makedirs(kaggle_dir, exist_ok=True)
with open(kaggle_json, "w") as f:
    json.dump({"username": username, "key": api_key}, f)

os.chmod(kaggle_json, 0o600)
print(f"\n[OK] Token saved to {kaggle_json}")
print("     Run  python scripts/download_kaggle_datasets.py  next.")
