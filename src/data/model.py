from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, dtype: torch.dtype = torch.float16, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    mdl.to(device)
    mdl.eval()
    return mdl, tok
