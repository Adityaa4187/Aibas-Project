import os
import requests
import pandas as pd
from bs4 import BeautifulSoup  

# DATASET URL 
url = "https://github.com/Adityaa4187/Aibas-Project/blob/main/WA_Fn-UseC_-HR-Employee-Attrition.csv"

# SAVE PATH (STRUCTURE FIXED)
SAVE_DIR = os.path.join("data", "raw")
SAVE_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)


def github_blob_to_raw(blob_url: str) -> str:
    """
    Converts GitHub blob URL -> raw URL
    """
    if "github.com" in blob_url and "/blob/" in blob_url:
        blob_url = blob_url.replace("github.com/", "raw.githubusercontent.com/")
        blob_url = blob_url.replace("/blob/", "/")
    return blob_url


def download_dataset():
    os.makedirs(SAVE_DIR, exist_ok=True)

    raw_url = github_blob_to_raw(url)

    print("[INFO] Downloading dataset from GitHub...")
    print(f"[INFO] URL: {raw_url}")
    print(f"[INFO] Saving to: {SAVE_PATH}")

    resp = requests.get(raw_url, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Download failed. HTTP {resp.status_code}\n"
            f"Response snippet: {resp.text[:200]}"
        )

    # Save content directly (CSV)
    with open(SAVE_PATH, "wb") as f:
        f.write(resp.content)

    # Quick verification load (ensures it's valid CSV)
    df = pd.read_csv(SAVE_PATH)
    print("[SUCCESS] Dataset downloaded and verified as CSV.")
    print("[INFO] Shape:", df.shape)
    print("[INFO] Saved at:", SAVE_PATH)
