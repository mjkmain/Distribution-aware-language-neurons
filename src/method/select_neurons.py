from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch

from src.method.overlap import load_hist_cache, pairwise_overlap
from src.utils import RESULTS_ROOT, cache_dir_for, model_short_name


def enumerate_bipartitions(L: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    # non-trivial (A, B) with A the smaller side; even-split ties keep index 0 in A
    out = []
    for size_a in range(1, L // 2 + 1):
        for A in combinations(range(L), size_a):
            if size_a * 2 == L and 0 not in A:
                continue
            out.append((A, tuple(i for i in range(L) if i not in A)))
    return out


def best_bipartition(overlap: torch.Tensor):
    # per neuron, the bipartition minimizing max_{k in A, k' in B} S[k, k']
    # (equivalent to single-linkage clustering with K=2)
    n_l, n_i, L, _ = overlap.shape
    parts = enumerate_bipartitions(L)
    cross_max = torch.empty(n_l, n_i, len(parts), dtype=overlap.dtype)
    for p, (A, B) in enumerate(parts):
        sub = overlap.index_select(-2, torch.tensor(A)).index_select(-1, torch.tensor(B))
        cross_max[..., p] = sub.amax(dim=(-2, -1))
    best_p = cross_max.argmin(dim=-1)
    best_cost = cross_max.gather(-1, best_p.unsqueeze(-1)).squeeze(-1)
    return parts, best_p, best_cost


def pair_overlap_threshold(overlap: torch.Tensor, pct: float) -> float:
    # tau = bottom-P percentile of all off-diagonal pair overlaps
    L = overlap.shape[-1]
    iu, ju = torch.triu_indices(L, L, offset=1)
    vals = overlap[:, :, iu, ju].reshape(-1).contiguous()
    k = max(1, int(vals.numel() * pct / 100.0))
    return float(vals.kthvalue(k).values.item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pct", type=float, default=1.0, help="bottom percentile P, in percent")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    bundle = load_hist_cache(cache_dir_for(args.model) / "distribution")
    languages = bundle["languages"]
    print(f"languages: {languages}, hist shape: {tuple(bundle['hist'].shape)}")

    overlap = pairwise_overlap(bundle["hist"])
    tau = pair_overlap_threshold(overlap, args.pct)
    parts, best_p, best_cost = best_bipartition(overlap)

    neurons = []
    for li, ni in (best_cost <= tau).nonzero(as_tuple=False).tolist():
        A, _ = parts[int(best_p[li, ni])]
        neurons.append({
            "layer": li,
            "neuron": ni,
            "cluster": [languages[i] for i in A],
            "cost": float(best_cost[li, ni]),
        })
    neurons.sort(key=lambda e: (len(e["cluster"]), e["cost"]))
    n_sln = sum(1 for e in neurons if len(e["cluster"]) == 1)
    print(f"P={args.pct}%: tau={tau:.4f}, {len(neurons)} neurons "
          f"({n_sln} SLN + {len(neurons) - n_sln} MLN)")

    out_path = args.out or (RESULTS_ROOT / model_short_name(args.model) / "populations.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "languages": languages,
        "pct": args.pct,
        "tau": tau,
        "neurons": neurons,
    }, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
