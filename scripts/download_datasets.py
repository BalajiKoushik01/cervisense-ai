"""
CerviSense-AI Dataset Downloader
Downloads SIPaKMeD and AnnoCerv to data/raw/.
CRIC Cervix is downloaded separately via browser.
Intel MobileODT requires Kaggle login (handled via browser).

Run from the cervisense-ai root:
    python scripts/download_datasets.py
"""

import os
import sys
import urllib.request
import subprocess
import zipfile
import shutil

try:
    import py7zr
except ImportError:
    print("[INFO] py7zr not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "py7zr"])
    import py7zr

try:
    import requests
except ImportError:
    print("[INFO] requests not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# ─────────────────────────────────────────────────────────────────────────────
# SIPaKMeD – direct from University of Ioannina
# 5 classes: Superficial-Intermediate, Parabasal, Koilocytotic,
#             Dyskeratotic, Metaplastic
# ─────────────────────────────────────────────────────────────────────────────
SIPAKMED_URLS = {
    "im_Superficial-Intermediate.7z": (
        "https://www.cse.uoi.gr/~marina/./SIPAKMED/im_Superficial-Intermediate.7z"
    ),
    "im_Parabasal.7z": (
        "https://www.cse.uoi.gr/~marina/./SIPAKMED/im_Parabasal.7z"
    ),
    "im_Koilocytotic.7z": (
        "https://www.cse.uoi.gr/~marina/./SIPAKMED/im_Koilocytotic.7z"
    ),
    "im_Metaplastic.7z": (
        "https://www.cse.uoi.gr/~marina/./SIPAKMED/im_Metaplastic.7z"
    ),
    "im_Dyskeratotic.7z": (
        "https://www.cse.uoi.gr/~marina/./SIPAKMED/im_Dyskeratotic.7z"
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# AnnoCerv – public GitHub release (colposcopy images)
# ─────────────────────────────────────────────────────────────────────────────
ANNOCERV_REPO = "https://github.com/iclx/AnnoCerv"


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def download_file(url, dest_path, chunk_size=1024 * 512, max_retries=5):
    """Stream download with HTTP Range-based resume and retry on failure."""
    headers = {"User-Agent": "Mozilla/5.0 (CerviSense-AI Dataset Downloader)"}

    for attempt in range(1, max_retries + 1):
        # Determine how much we already have
        existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

        # First, get Content-Length to see if already complete
        head = requests.head(url, headers=headers, timeout=30, allow_redirects=True)
        total = int(head.headers.get("content-length", 0))
        if total and existing_size >= total:
            print(f"  [SKIP] Already downloaded: {os.path.basename(dest_path)}")
            return

        if existing_size:
            print(f"  [RESUME] {os.path.basename(dest_path)} ({existing_size//(1024*1024)} MB already, attempt {attempt})")
            req_headers = {**headers, "Range": f"bytes={existing_size}-"}
        else:
            print(f"  [DL]   {url}  (attempt {attempt})")
            req_headers = headers

        try:
            resp = requests.get(url, stream=True, headers=req_headers, timeout=120)
            resp.raise_for_status()
            mode = "ab" if existing_size else "wb"
            downloaded = existing_size
            with open(dest_path, mode) as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            print(f"\r    {pct}% ({downloaded//(1024*1024)} MB / {total//(1024*1024)} MB)", end="")
            print()
            print(f"  [OK]   Saved to {dest_path}")
            return
        except Exception as exc:
            print(f"\n  [WARN] Attempt {attempt} failed: {exc}")
            if attempt == max_retries:
                raise
            import time
            time.sleep(3 * attempt)

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")


def extract_7z(archive_path, dest_dir):
    """Extract a .7z archive."""
    print(f"  [EX]   Extracting {os.path.basename(archive_path)} ...")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=dest_dir)
    print(f"  [OK]   Extracted to {dest_dir}")


def extract_zip(archive_path, dest_dir):
    """Extract a .zip archive."""
    print(f"  [EX]   Extracting {os.path.basename(archive_path)} ...")
    with zipfile.ZipFile(archive_path, "r") as z:
        z.extractall(dest_dir)
    print(f"  [OK]   Extracted to {dest_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1: SIPaKMeD
# ─────────────────────────────────────────────────────────────────────────────
def download_sipakmed():
    print("\n" + "=" * 60)
    print("DATASET 1: SIPaKMeD (University of Ioannina)")
    print("=" * 60)

    sipakmed_dir = os.path.join(RAW_DIR, "sipakmed")
    archive_dir = os.path.join(sipakmed_dir, "_archives")
    make_dirs(sipakmed_dir, archive_dir)

    for filename, url in SIPAKMED_URLS.items():
        archive_path = os.path.join(archive_dir, filename)
        class_name = filename.replace("im_", "").replace(".7z", "")
        class_out_dir = os.path.join(sipakmed_dir, class_name)

        if os.path.exists(class_out_dir) and os.listdir(class_out_dir):
            print(f"  [SKIP] {class_name} already extracted.")
            continue

        download_file(url, archive_path)
        extract_7z(archive_path, sipakmed_dir)

    print("[DONE] SIPaKMeD download complete.")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2: AnnoCerv (GitHub)
# ─────────────────────────────────────────────────────────────────────────────
def download_annocerv():
    print("\n" + "=" * 60)
    print("DATASET 2: AnnoCerv (GitHub)")
    print("=" * 60)

    annocerv_dir = os.path.join(RAW_DIR, "annocerv")

    if os.path.exists(os.path.join(annocerv_dir, ".git")):
        print("  [SKIP] AnnoCerv repo already cloned.")
        return

    # Check if git is installed
    git_cmd = shutil.which("git")
    if git_cmd is None:
        print("  [WARN] git not found. Trying ZIP download fallback...")
        zip_url = "https://github.com/iclx/AnnoCerv/archive/refs/heads/main.zip"
        zip_path = os.path.join(RAW_DIR, "annocerv.zip")
        download_file(zip_url, zip_path)
        extract_zip(zip_path, RAW_DIR)
        # Rename extracted folder to 'annocerv'
        extracted = os.path.join(RAW_DIR, "AnnoCerv-main")
        if os.path.exists(extracted):
            shutil.move(extracted, annocerv_dir)
        return

    print(f"  [GIT]  Cloning AnnoCerv from {ANNOCERV_REPO} ...")
    subprocess.check_call(
        [git_cmd, "clone", "--depth", "1", ANNOCERV_REPO, annocerv_dir]
    )
    print("[DONE] AnnoCerv download complete.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("CerviSense-AI Dataset Downloader")
    print(f"Root directory : {BASE_DIR}")
    print(f"Target RAW dir : {RAW_DIR}")
    make_dirs(RAW_DIR)

    download_sipakmed()
    download_annocerv()

    print("\n" + "=" * 60)
    print("DATASET 3: CRIC Cervix")
    print("=" * 60)
    print("  [INFO] CRIC Cervix is downloaded via browser (Figshare collection).")
    print("  [INFO] The browser task will handle this automatically.")

    print("\n" + "=" * 60)
    print("DATASET 4: Intel MobileODT")
    print("=" * 60)
    print("  [INFO] Intel MobileODT requires Kaggle login.")
    print("  [INFO] The browser task will handle this automatically.")

    print("\n[ALL DONE] Scripted downloads finished.")
    print("Next step: run `python data_utils/preprocess.py` once all datasets are in place.")


if __name__ == "__main__":
    main()
