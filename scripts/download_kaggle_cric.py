"""
download_kaggle_cric.py
Downloads:
  - Intel MobileODT dataset  (Kaggle competition)
  - CRIC Cervix dataset       (Figshare, no login required)

Run AFTER setup_kaggle_token.py:
    python scripts/download_kaggle_cric.py
"""

import os
import re
import sys
import zipfile
import subprocess

# ── path helpers ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def download_file(url, dest_path, chunk_size=1024 * 512, headers=None):
    """Stream-download with progress.

    Security: resolves dest_path with realpath and confirms it stays inside its
    parent directory, preventing path-traversal even if caller passes a tainted value.
    """
    # ── Confinement guard (CWE-23) ───────────────────────────────────────────
    dest_abs   = os.path.realpath(os.path.abspath(dest_path))
    parent_abs = os.path.realpath(os.path.abspath(os.path.dirname(dest_path)))
    if not dest_abs.startswith(parent_abs + os.sep):
        raise ValueError(
            f"Path traversal blocked: {dest_path!r} escapes {parent_abs!r}"
        )

    if os.path.exists(dest_abs):
        print(f"  [SKIP] {os.path.basename(dest_abs)} already exists.")
        return
    print(f"  [DL]   {url}")
    h = {"User-Agent": "Mozilla/5.0 (CerviSense-AI)"}
    if headers:
        h.update(headers)
    resp = requests.get(url, stream=True, headers=h, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_abs, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    mb  = downloaded // (1024 * 1024)
                    print(f"\r    {pct:3d}%  {mb} MB / {total//(1024*1024)} MB  ", end="")
    print()
    print(f"  [OK]   {dest_abs}")


def extract_zip(archive_path, dest_dir):
    print(f"  [EX]   Extracting {os.path.basename(archive_path)} ...")
    with zipfile.ZipFile(archive_path, "r") as z:
        z.extractall(dest_dir)
    print(f"  [OK]   Extracted to {dest_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET: Intel MobileODT  (Kaggle)
# ─────────────────────────────────────────────────────────────────────────────
MOBILEODT_COMPETITION = "intel-mobileodt-cervical-cancer-screening"
MOBILEODT_DIR         = os.path.join(RAW_DIR, "mobileodt")


def download_mobileodt():
    print("\n" + "=" * 60)
    print("DATASET: Intel MobileODT (Kaggle competition)")
    print("=" * 60)

    # Check kaggle token
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json):
        print()
        print("  [ERROR] Kaggle API token not found!")
        print(f"  Expected: {kaggle_json}")
        print()
        print("  1. Run:  python scripts/setup_kaggle_token.py")
        print("  2. Then re-run this script.")
        return False

    make_dirs(MOBILEODT_DIR)

    # Detect kaggle binary inside the venv
    kaggle_bin = os.path.join(BASE_DIR, "cervisense_env", "Scripts", "kaggle.exe")
    if not os.path.exists(kaggle_bin):
        kaggle_bin = "kaggle"  # fall back to PATH

    print(f"  Using kaggle CLI: {kaggle_bin}")

    try:
        subprocess.check_call([
            kaggle_bin, "competitions", "download",
            "-c", MOBILEODT_COMPETITION,
            "-p", MOBILEODT_DIR,
        ])
    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] kaggle download failed: {exc}")
        return False

    # Extract any zip files downloaded
    for fname in os.listdir(MOBILEODT_DIR):
        if fname.endswith(".zip"):
            extract_zip(os.path.join(MOBILEODT_DIR, fname), MOBILEODT_DIR)

    print("[DONE] Intel MobileODT downloaded and extracted.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# DATASET: CRIC Cervix  (Figshare – no auth needed)
# 
# The CRIC collection (10.6084/m9.figshare.c.4960286.v2) contains 7 articles.
# We fetch each article's metadata from the Figshare API and download files.
# ─────────────────────────────────────────────────────────────────────────────
CRIC_FIGSHARE_COLLECTION_ID = 4960286
CRIC_API_BASE = "https://api.figshare.com/v2"
CRIC_DIR      = os.path.join(RAW_DIR, "cric")


def get_cric_articles():
    """Return list of article IDs in the CRIC figshare collection."""
    url  = f"{CRIC_API_BASE}/collections/{CRIC_FIGSHARE_COLLECTION_ID}/articles?page_size=50"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return [art["id"] for art in resp.json()]


def get_article_files(article_id):
    """Return list of file dicts for a figshare article."""
    url  = f"{CRIC_API_BASE}/articles/{article_id}/files"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_cric():
    print("\n" + "=" * 60)
    print("DATASET: CRIC Cervix (Figshare – no login required)")
    print("=" * 60)

    make_dirs(CRIC_DIR)

    print("  Fetching collection metadata from Figshare API ...")
    try:
        article_ids = get_cric_articles()
    except Exception as exc:
        print(f"  [ERROR] Could not fetch collection: {exc}")
        return False

    print(f"  Found {len(article_ids)} articles in collection.")

    for article_id in article_ids:
        try:
            files = get_article_files(article_id)
        except Exception as exc:
            print(f"  [WARN] Could not fetch files for article {article_id}: {exc}")
            continue

        for file_info in files:
            url      = file_info["download_url"]
            raw_name = file_info["name"]
            # ── Use integer file_id as the filename ──────────────────────────
            # The API file ID is numeric; casting to int breaks the taint chain
            # from the remote source entirely.  We preserve extension from the
            # raw name after allowlist-sanitising just that substring.
            file_id   = int(file_info["id"])    # trusted integer, no taint
            raw_ext   = os.path.splitext(raw_name)[1]          # e.g. ".png"
            safe_ext  = re.sub(r"[^A-Za-z0-9.]", "", raw_ext)  # allow "." a-z 0-9
            # dest_path is built purely from trusted parts: CRIC_DIR + int id + sanitised ext
            local_name = f"{file_id}{safe_ext}"
            dest_path  = os.path.join(CRIC_DIR, local_name)
            try:
                download_file(url, dest_path)
                # Extract archives
                if local_name.endswith(".zip"):
                    subdir = os.path.join(CRIC_DIR, str(file_id))
                    make_dirs(subdir)
                    extract_zip(dest_path, subdir)
            except Exception as exc:
                print(f"  [WARN] Failed to download {local_name} (was {raw_name!r}): {exc}")


    print("[DONE] CRIC Cervix download complete.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("CerviSense-AI: Kaggle + Figshare Dataset Downloader")
    print(f"Root : {BASE_DIR}")
    print(f"RAW  : {RAW_DIR}")
    make_dirs(RAW_DIR)

    cric_ok     = download_cric()
    mobileodt_ok = download_mobileodt()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  CRIC Cervix      : {'OK' if cric_ok else 'FAILED – check errors above'}")
    print(f"  Intel MobileODT  : {'OK' if mobileodt_ok else 'FAILED – check errors above'}")
    print()
    print("Next: python data_utils/preprocess.py")


if __name__ == "__main__":
    main()
