from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn
from tqdm import tqdm

# callback: (layer_idx, a, u) -> (a, u), where a = act_fn(gate_proj(x)) and
# u = up_proj(x), both of shape (batch, seq, d_ffn)
Callback = Callable[[int, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class GatedMLPWrapper:
    def __init__(self, mlp: nn.Module, layer_idx: int, callback: Callback | None):
        self.mlp = mlp
        self.layer_idx = layer_idx
        self.callback = callback
        self._orig_forward = mlp.forward
        mlp.forward = self._wrapped_forward

    def _wrapped_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.mlp.gate_proj(x)
        u = self.mlp.up_proj(x)
        a = self.mlp.act_fn(gate)
        if self.callback is not None:
            a, u = self.callback(self.layer_idx, a, u)
        return self.mlp.down_proj(a * u)

    def remove(self) -> None:
        self.mlp.forward = self._orig_forward


def get_mlps(model: nn.Module) -> list[nn.Module]:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError(f"cannot locate transformer layers on {type(model).__name__}")
    return [layer.mlp for layer in layers]


@contextmanager
def install_callback(model: nn.Module, callback: Callback | None):
    wrappers = [GatedMLPWrapper(mlp, i, callback) for i, mlp in enumerate(get_mlps(model))]
    try:
        yield wrappers
    finally:
        for w in wrappers:
            w.remove()


class SumAccum:
    def __init__(self, n_layers: int, n_intermediate: int, device: torch.device):
        self.sum = torch.zeros(n_layers, n_intermediate, dtype=torch.float32, device=device)
        self.sum_sq = torch.zeros_like(self.sum)
        self.count = 0
        self._mask: torch.Tensor | None = None

    def set_mask(self, attn_mask: torch.Tensor) -> None:
        m = attn_mask.bool()
        self._mask = m
        self.count += int(m.sum().item())

    def __call__(self, layer_idx: int, a: torch.Tensor, u: torch.Tensor):
        a_f = a.float()
        if self._mask is None:
            s = a_f.sum(dim=(0, 1))
            sq = (a_f * a_f).sum(dim=(0, 1))
        else:
            mb = self._mask.to(a.device).unsqueeze(-1).to(a_f.dtype)
            s = (a_f * mb).sum(dim=(0, 1))
            sq = (a_f * a_f * mb).sum(dim=(0, 1))
        self.sum[layer_idx] += s.to(self.sum.device)
        self.sum_sq[layer_idx] += sq.to(self.sum.device)
        return a, u


def tokenize_concat(tokenizer, texts: list[str], max_length: int, target_tokens: int):
    # concatenate token streams and emit (1, <=max_length) chunks until target_tokens
    pending: list[int] = []
    emitted = 0
    for txt in texts:
        if emitted >= target_tokens:
            break
        pending.extend(tokenizer(txt, add_special_tokens=False)["input_ids"])
        while len(pending) >= max_length and emitted < target_tokens:
            chunk, pending = pending[:max_length], pending[max_length:]
            t = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
            yield t, torch.ones_like(t)
            emitted += max_length
    if emitted < target_tokens and pending:
        t = torch.tensor(pending, dtype=torch.long).unsqueeze(0)
        yield t, torch.ones_like(t)


def stream_forward(model, tokenizer, texts, *, target_tokens, max_length, callback) -> int:
    device = next(model.parameters()).device
    n_seen = 0
    pbar = tqdm(total=target_tokens, unit="tok", leave=False)
    with torch.inference_mode(), install_callback(model, callback):
        for ids, attn in tokenize_concat(tokenizer, texts, max_length, target_tokens):
            ids, attn = ids.to(device), attn.to(device)
            callback.set_mask(attn)
            model(input_ids=ids, attention_mask=attn)
            inc = int(attn.sum().item())
            n_seen += inc
            pbar.update(inc)
            if n_seen >= target_tokens:
                break
    pbar.close()
    return n_seen
