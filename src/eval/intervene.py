from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Iterable

import torch

from src.data.extraction import get_mlps, install_callback


class MeanPatchIntervention:
    # replace a[..., n] with a fixed per-(layer, neuron) scalar during forward
    def __init__(self, replacements: Iterable[tuple[int, int, float]],
                 n_layers: int, n_intermediate: int):
        by_layer: dict[int, dict[int, float]] = defaultdict(dict)
        for layer, neuron, value in replacements:
            li, ni = int(layer), int(neuron)
            if not (0 <= li < n_layers and 0 <= ni < n_intermediate):
                raise ValueError(f"index out of range: layer={li}, neuron={ni}")
            by_layer[li][ni] = float(value)
        self.by_layer_idx = {li: torch.tensor(sorted(m), dtype=torch.long)
                             for li, m in by_layer.items()}
        self.by_layer_val = {li: torch.tensor([by_layer[li][n] for n in sorted(m)],
                                              dtype=torch.float32)
                             for li, m in by_layer.items()}
        self._dev_cache: dict = {}

    def __call__(self, layer_idx: int, a: torch.Tensor, u: torch.Tensor):
        ns = self.by_layer_idx.get(layer_idx)
        if ns is None:
            return a, u
        key = (layer_idx, a.device, a.dtype)
        if key not in self._dev_cache:
            self._dev_cache[key] = (
                ns.to(a.device),
                self.by_layer_val[layer_idx].to(device=a.device, dtype=a.dtype),
            )
        ns_dev, vals_dev = self._dev_cache[key]
        a = a.clone()
        a[..., ns_dev] = vals_dev
        return a, u


@contextmanager
def mean_patched_neurons(model, replacements: Iterable[tuple[int, int, float]]):
    mlps = get_mlps(model)
    cb = MeanPatchIntervention(replacements, n_layers=len(mlps),
                               n_intermediate=int(mlps[0].gate_proj.out_features))
    with install_callback(model, cb):
        yield cb
