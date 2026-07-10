from __future__ import annotations

from pathlib import Path

import torch


def load_hist_cache(cache_dir: Path) -> dict:
    meta = torch.load(cache_dir / "meta.pt", map_location="cpu", weights_only=False)
    languages = list(meta["languages"])
    hist = torch.stack(
        [torch.load(cache_dir / f"hist_{lg}.pt", map_location="cpu",
                    weights_only=False)["hist"].to(torch.int64) for lg in languages],
        dim=2,
    )  # (n_layers, n_intermediate, L, n_bins+2)
    return {"meta": meta, "hist": hist, "languages": languages}


def hist_to_pmf(hist: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    s = hist.to(torch.float64).sum(dim=-1, keepdim=True).clamp_min(eps)
    return hist.to(torch.float64) / s


def pairwise_overlap(hist: torch.Tensor, chunk_layers: int = 1) -> torch.Tensor:
    # S[k, k'] = sum_b min(P_b, Q_b), computed per layer chunk to bound memory
    n_l, n_i, L, _ = hist.shape
    out = torch.zeros(n_l, n_i, L, L, dtype=torch.float32)
    for s in range(0, n_l, chunk_layers):
        e = min(s + chunk_layers, n_l)
        pmf = hist_to_pmf(hist[s:e])
        ovl = torch.minimum(pmf.unsqueeze(-3), pmf.unsqueeze(-2)).sum(dim=-1)
        out[s:e] = ovl.to(torch.float32)
    return out
