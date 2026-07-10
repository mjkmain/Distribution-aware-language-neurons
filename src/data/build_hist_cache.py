from __future__ import annotations

import argparse
import json

import torch

from src.data.corpora import LANGS, wiki_texts_for_tokens
from src.data.extraction import SumAccum, stream_forward
from src.data.model import load_model
from src.utils import auto_device, cache_dir_for, set_seed

# SiLU floor is about -0.2785; pad so it stays strictly above the lowest inner bin edge
SILU_FLOOR_PAD = -0.4


class HistAccum:
    # per-(layer, neuron) histogram; bin 0 = underflow, 1..n_bins inner, n_bins+1 = overflow
    def __init__(self, bin_lo: torch.Tensor, bin_hi: torch.Tensor, n_bins: int, device: torch.device):
        self.bin_lo = bin_lo.to(device)
        self.bin_width = ((bin_hi.to(device) - self.bin_lo) / float(n_bins)).clamp_min(1e-6)
        self.n_bins = n_bins
        self.hist = torch.zeros(*bin_lo.shape, n_bins + 2, dtype=torch.int32, device=device)
        self.count = 0
        self._mask: torch.Tensor | None = None

    def set_mask(self, attn_mask: torch.Tensor) -> None:
        m = attn_mask.bool()
        self._mask = m
        self.count += int(m.sum().item())

    def __call__(self, layer_idx: int, a: torch.Tensor, u: torch.Tensor):
        if self._mask is None:
            flat_a = a.reshape(-1, a.shape[-1])
        else:
            flat_a = a[self._mask.to(a.device)]
        if flat_a.numel() == 0:
            return a, u
        flat_a = flat_a.float().to(self.hist.device)
        idx = torch.div(flat_a - self.bin_lo[layer_idx], self.bin_width[layer_idx],
                        rounding_mode="floor").to(torch.long)
        idx = (idx + 1).clamp(0, self.n_bins + 1)
        idx_t = idx.transpose(0, 1).contiguous()
        self.hist[layer_idx].scatter_add_(1, idx_t, torch.ones_like(idx_t, dtype=torch.int32))
        return a, u


def run_pass1(model, tokenizer, languages, *, target_tokens, max_length, out_path, overwrite):
    if out_path.exists() and not overwrite:
        print(f"[pass1] {out_path.name} exists, skipping")
        return torch.load(out_path, map_location="cpu", weights_only=False)

    n_layers = model.config.num_hidden_layers
    n_intermediate = model.config.intermediate_size
    accum = SumAccum(n_layers, n_intermediate, device=next(model.parameters()).device)
    for lang in languages:
        texts = wiki_texts_for_tokens(lang, source="calib", target_tokens=target_tokens)
        n = stream_forward(model, tokenizer, texts, target_tokens=target_tokens,
                           max_length=max_length, callback=accum)
        print(f"[pass1] {lang}: {n} tokens")

    mean = (accum.sum / float(accum.count)).cpu()
    std = ((accum.sum_sq / float(accum.count)).cpu() - mean * mean).clamp_min(0).sqrt()
    payload = {
        "mean": mean, "std": std, "count": int(accum.count),
        "n_layers": int(n_layers), "n_intermediate": int(n_intermediate),
        "languages": list(languages),
    }
    torch.save(payload, out_path)
    print(f"[pass1] wrote {out_path}")
    return payload


def run_pass2(model, tokenizer, languages, *, target_tokens, max_length, n_bins,
              meta_path, out_dir, overwrite):
    if not meta_path.exists():
        raise SystemExit(f"pass-1 meta missing: {meta_path}")
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    mean, std = meta["mean"], meta["std"]

    # asymmetric range: SiLU has a hard left floor but a heavy right tail
    bin_lo = torch.maximum(mean - 3.0 * std, torch.full_like(mean, SILU_FLOOR_PAD))
    bin_hi = mean + 6.0 * std
    flat = (bin_hi - bin_lo) < 1e-3
    if flat.any():
        bin_lo = torch.where(flat, mean - 0.05, bin_lo)
        bin_hi = torch.where(flat, mean + 0.05, bin_hi)
    meta.update({"bin_lo": bin_lo, "bin_hi": bin_hi, "n_bins": int(n_bins)})
    torch.save(meta, meta_path)

    device = next(model.parameters()).device
    for lang in languages:
        out_path = out_dir / f"hist_{lang}.pt"
        if out_path.exists() and not overwrite:
            print(f"[pass2] {out_path.name} exists, skipping")
            continue
        texts = wiki_texts_for_tokens(lang, source="calib", target_tokens=target_tokens)
        accum = HistAccum(bin_lo, bin_hi, n_bins=n_bins, device=device)
        n = stream_forward(model, tokenizer, texts, target_tokens=target_tokens,
                           max_length=max_length, callback=accum)
        torch.save({"lang": lang, "n_tokens": int(n), "hist": accum.hist.cpu(),
                    "n_bins": int(n_bins)}, out_path)
        print(f"[pass2] {lang}: {n} tokens -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--languages", nargs="+", default=LANGS)
    p.add_argument("--target-tokens", type=int, default=100_000)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--n-bins", type=int, default=400)
    p.add_argument("--device", default=auto_device())
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--phase", default="both", choices=["pass1", "pass2", "both"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    out_dir = cache_dir_for(args.model) / "distribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.pt"

    model, tok = load_model(args.model, dtype=dtype, device=args.device)

    if args.phase in ("pass1", "both"):
        run_pass1(model, tok, args.languages, target_tokens=args.target_tokens,
                  max_length=args.max_length, out_path=meta_path, overwrite=args.overwrite)
    if args.phase in ("pass2", "both"):
        run_pass2(model, tok, args.languages, target_tokens=args.target_tokens,
                  max_length=args.max_length, n_bins=args.n_bins,
                  meta_path=meta_path, out_dir=out_dir, overwrite=args.overwrite)

    (out_dir / "manifest.json").write_text(json.dumps({
        "model": args.model, "languages": list(args.languages),
        "target_tokens": args.target_tokens, "max_length": args.max_length,
        "n_bins": args.n_bins, "dtype": args.dtype,
    }, indent=2))


if __name__ == "__main__":
    main()
