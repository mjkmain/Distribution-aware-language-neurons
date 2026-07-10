from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.data.corpora import SHORT, load_flores, wiki_texts_for_tokens
from src.data.model import load_model
from src.eval.intervene import mean_patched_neurons
from src.eval.ppl import PplResult, corpus_ppl_concat
from src.utils import RESULTS_ROOT, cache_dir_for, model_short_name, set_seed

INV_SHORT = {v: k for k, v in SHORT.items()}


def _build_groups(doc: dict, neuron_type: str, groups_arg: list[str] | None,
                  n_groups: int) -> dict[tuple[str, ...], list[tuple[int, int]]]:
    # SLN: one group per target language; MLN: one group per cluster A
    by_cluster: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    for e in doc["neurons"]:
        by_cluster[tuple(e["cluster"])].append((e["layer"], e["neuron"]))

    if neuron_type == "sln":
        return {(lg,): by_cluster[(lg,)] for lg in doc["languages"] if (lg,) in by_cluster}

    mln = {c: ns for c, ns in by_cluster.items() if len(c) > 1}
    if groups_arg:
        chosen = {}
        for g in groups_arg:
            cluster = tuple(sorted((INV_SHORT[s] for s in g.split(",")),
                                   key=doc["languages"].index))
            chosen[cluster] = mln.get(cluster, [])
        return chosen
    # default: the most-populated clusters
    top = sorted(mln.items(), key=lambda kv: -len(kv[1]))[:n_groups]
    return dict(top)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--neurons", required=True, choices=("sln", "mln"))
    p.add_argument("--corpus", default="flores_devtest", choices=("flores_devtest", "wiki_test"))
    p.add_argument("--populations", type=Path, default=None)
    p.add_argument("--groups", nargs="+", default=None,
                   help="MLN clusters as comma-joined short codes, e.g. zh,ja fr,es")
    p.add_argument("--n-groups", type=int, default=4,
                   help="number of most-populated MLN clusters when --groups is not given")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-test-tokens", type=int, default=1_000_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    pops_path = args.populations or (RESULTS_ROOT / model_short_name(args.model) / "populations.json")
    doc = json.loads(pops_path.read_text())
    languages = list(doc["languages"])

    groups = _build_groups(doc, args.neurons, args.groups, args.n_groups)
    for cluster, ns in groups.items():
        print(f"group {{{','.join(SHORT[lg] for lg in cluster)}}}: {len(ns)} neurons")

    mean_path = cache_dir_for(args.model) / "distribution" / "activation_means.pt"
    mp = torch.load(mean_path, map_location="cpu", weights_only=False)
    mean_tensor = mp["mean"].numpy()
    if list(mp["languages"]) != languages:
        idx = [list(mp["languages"]).index(lg) for lg in languages]
        mean_tensor = mean_tensor[..., idx]

    model, tokenizer = load_model(args.model, dtype=getattr(torch, args.dtype),
                                  device=args.device)

    n_test_tokens = args.n_test_tokens if args.n_test_tokens > 0 else None
    if args.corpus == "flores_devtest":
        texts_per_lang = load_flores(languages, split="devtest")
    else:
        texts_per_lang = {
            lg: wiki_texts_for_tokens(lg, source="test", target_tokens=n_test_tokens or 1_000_000)
            for lg in languages
        }

    def _ppl(texts) -> PplResult:
        return corpus_ppl_concat(model, tokenizer, texts, max_length=args.max_length,
                                 batch_size=args.batch_size, target_tokens=n_test_tokens)

    clean: dict[str, PplResult] = {}
    for lg in languages:
        clean[lg] = _ppl(texts_per_lang[lg])
        print(f"clean {lg}: ppl={clean[lg].ppl:.4f} ({clean[lg].n_tokens} tokens)")

    rows = []
    for cluster, neurons in groups.items():
        if not neurons:
            print(f"group {{{','.join(SHORT[lg] for lg in cluster)}}} is empty, skipping")
            continue
        # replacement = mean activation over the complement languages (k not in A)
        mask = np.array([lg not in cluster for lg in languages])
        replacements = [(li, ni, float(mean_tensor[li, ni][mask].mean()))
                        for (li, ni) in neurons]
        target = ",".join(SHORT[lg] for lg in cluster)
        t0 = time.time()
        with mean_patched_neurons(model, replacements):
            for eval_lang in languages:
                res = _ppl(texts_per_lang[eval_lang])
                clean_nll = clean[eval_lang].nll_sum / clean[eval_lang].n_tokens
                intervened_nll = res.nll_sum / res.n_tokens
                rows.append({
                    "target": target,
                    "eval_lang": SHORT[eval_lang],
                    "n_neurons": len(neurons),
                    "clean_ppl": clean[eval_lang].ppl,
                    "intervened_ppl": res.ppl,
                    "clean_nll_per_token": clean_nll,
                    "intervened_nll_per_token": intervened_nll,
                    "delta_nll": intervened_nll - clean_nll,
                })
        print(f"target {{{target}}} ({len(neurons)} neurons): "
              f"{len(languages)} evals in {time.time() - t0:.1f}s")

    if not rows:
        raise SystemExit("no non-empty groups to evaluate")
    out_path = args.output or (RESULTS_ROOT / model_short_name(args.model)
                               / f"selectivity_{args.neurons}_{args.corpus}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
