# ============================================================
# scripts/download_data.py — Phase 1 (HPC)
# Run this ONCE on the Stanage LOGIN node (needs internet access,
# which compute nodes typically don't have). Pre-caches the chosen
# model so later sbatch jobs can load it from local disk/cache
# without needing network access.
# ============================================================

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_TO_DOWNLOAD = "Qwen/Qwen2.5-7B-Instruct"  # or whichever size you decide on


def predownload_model(model_name: str):
    print(f"Pre-downloading/caching model: {model_name} ...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
    print(f"  {model_name} cached successfully.")


if __name__ == "__main__":
    predownload_model(MODEL_TO_DOWNLOAD)