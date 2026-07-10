from __future__ import annotations

import argparse

import torch

from src.data.corpora import LANGS, wiki_texts_for_tokens
from src.data.extraction import SumAccum, stream_forward
from src.data.model import load_model
from src.utils import auto_device, cache_dir_for, set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--languages", nargs="+", default=LANGS)
    p.add_argument("--target-tokens", type=int, default=1_000_000)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default=auto_device())
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out_path = cache_dir_for(args.model) / "distribution" / "activation_means.pt"
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists, pass --overwrite to recompute")
        return

    model, tok = load_model(args.model, dtype=getattr(torch, args.dtype), device=args.device)
    n_layers = int(model.config.num_hidden_layers)
    n_intermediate = int(model.config.intermediate_size)
    device = next(model.parameters()).device

    mean = torch.zeros(n_layers, n_intermediate, len(args.languages), dtype=torch.float32)
    for k, lang in enumerate(args.languages):
        texts = wiki_texts_for_tokens(lang, source="calib", target_tokens=args.target_tokens)
        accum = SumAccum(n_layers, n_intermediate, device=device)
        n = stream_forward(model, tok, texts, target_tokens=args.target_tokens,
                           max_length=args.max_length, callback=accum)
        mean[..., k] = (accum.sum / float(accum.count)).cpu()
        print(f"{lang}: {n} tokens")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": args.model, "languages": list(args.languages), "mean": mean}, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
