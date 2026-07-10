from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "src" / "cache"
RESULTS_ROOT = PROJECT_ROOT / "results"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_short_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def cache_dir_for(model_id: str) -> Path:
    p = CACHE_ROOT / model_short_name(model_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
