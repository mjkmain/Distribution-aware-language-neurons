from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PplResult:
    ppl: float
    nll_sum: float
    n_tokens: int


@torch.no_grad()
def corpus_ppl_concat(model, tokenizer, texts: list[str], *,
                      max_length: int, batch_size: int,
                      target_tokens: int | None = None) -> PplResult:
    # join all texts, slide non-overlapping windows, accumulate token-level NLL (nats)
    device = next(model.parameters()).device
    joined = "\n\n".join(t for t in texts if t)
    ids = tokenizer(joined, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    n = ids.numel()
    if n < 2:
        return PplResult(ppl=float("nan"), nll_sum=0.0, n_tokens=0)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    nll_sum, n_scored = 0.0, 0
    starts = list(range(0, n, max_length))
    for batch_start in range(0, len(starts), batch_size):
        if target_tokens is not None and n_scored >= target_tokens:
            break
        windows = [ids[s: s + max_length] for s in starts[batch_start: batch_start + batch_size]]
        windows = [w for w in windows if w.numel() >= 2]
        if not windows:
            continue
        max_len = max(w.numel() for w in windows)
        input_ids = torch.full((len(windows), max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(windows), max_len), dtype=torch.long, device=device)
        for i, w in enumerate(windows):
            input_ids[i, : w.numel()] = w
            attn[i, : w.numel()] = 1

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attn[:, 1:].contiguous().to(torch.bool).view(-1)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            reduction="none",
        )
        nll_sum += float(loss[shift_mask].sum().item())
        n_scored += int(shift_mask.sum().item())

    if n_scored == 0:
        return PplResult(ppl=float("nan"), nll_sum=0.0, n_tokens=0)
    return PplResult(ppl=math.exp(nll_sum / n_scored), nll_sum=nll_sum, n_tokens=n_scored)
