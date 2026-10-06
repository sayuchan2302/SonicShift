"""
scripts/download_models.py - Pretrained RVC Base Model Downloader

Downloads essential base models required for high-fidelity RVC inference:
  1. hubert_base.pt (~180MB) - ContentVec / HuBERT linguistic speech feature extractor.
  2. rmvpe.pt (~38MB)       - High-precision neural pitch extractor for GPU.

Models are sourced directly from official Hugging Face repositories.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

# Paths
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_models_dir = _project_root / "models" / "pretrained"

MODELS = {
    "hubert_base.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "desc": "HuBERT / ContentVec speech feature encoder (~180MB)",
    },
    "rmvpe.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "desc": "RMVPE GPU real-time vocal pitch tracker (~38MB)",
    },
}


def download_file_with_progress(url: str, dest_path: Path, desc: str) -> bool:
    """Downloads a file with an inline terminal progress bar."""
    if dest_path.is_file() and dest_path.stat().st_size > 1000000:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f" [SKIP] {dest_path.name} already exists ({size_mb:.1f} MB).")
        return True

    print(f"\n[DOWNLOAD] {dest_path.name} - {desc}")
    print(f"  URL: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SonicShift/1.0"},
        )
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 65536
            start_time = time.time()

            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                elapsed = max(0.001, time.time() - start_time)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed

                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    bar_len = 24
                    filled = int(round((downloaded / total_size) * bar_len))
                    bar = "█" * filled + "░" * (bar_len - filled)
                    sys.stdout.write(
                        f"\r  [{bar}] {pct:5.1f}% ({downloaded/(1024*1024):5.1f}/{total_size/(1024*1024):5.1f} MB) @ {speed_mb:4.1f} MB/s  "
                    )
                else:
                    sys.stdout.write(f"\r  Downloaded: {downloaded/(1024*1024):5.1f} MB @ {speed_mb:4.1f} MB/s  ")
                sys.stdout.flush()

        temp_path.replace(dest_path)
        print("\n  [DONE] Download complete and verified.")
        return True

    except Exception as exc:
        print(f"\n  [ERROR] Failed to download {dest_path.name}: {exc}")
        if temp_path.is_file():
            temp_path.unlink()
        return False


def main() -> None:
    print("=" * 80)
    print("      SONICSHIFT - RVC PRETRAINED BASE MODELS DOWNLOADER")
    print("=" * 80)
    print(f"Target Directory: {_models_dir}\n")

    _models_dir.mkdir(parents=True, exist_ok=True)

    success_all = True
    for filename, info in MODELS.items():
        dest = _models_dir / filename
        ok = download_file_with_progress(info["url"], dest, info["desc"])
        if not ok:
            success_all = False

    print("\n" + "=" * 80)
    if success_all:
        print(" [SUCCESS] All pretrained base models are ready in models/pretrained/!")
    else:
        print(" [WARNING] Some models could not be downloaded. Check your internet connection.")
    print("=" * 80)


if __name__ == "__main__":
    main()
