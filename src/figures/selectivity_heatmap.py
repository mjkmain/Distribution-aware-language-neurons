from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.corpora import LANGS, SHORT
from src.utils import RESULTS_ROOT, model_short_name

LANG_ORDER = [SHORT[lg] for lg in LANGS]
CORPORA = [("wiki_test", "WIKI"), ("flores_devtest", "FLORES")]


def _load(csv_path: Path):
    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(index="target", columns="eval_lang", values="delta_nll",
                           aggfunc="mean")
    # rows in first-appearance order, columns in canonical language order
    row_order = list(dict.fromkeys(df["target"]))
    col_order = [lg for lg in LANG_ORDER if lg in pivot.columns]
    pivot = pivot.reindex(index=row_order, columns=col_order)
    n_neurons = int(df["n_neurons"].sum() / len(col_order))
    return pivot.to_numpy(dtype=float), row_order, col_order, n_neurons


def _draw(ax, mat, rows, cols, vmax, show_y):
    mat = np.maximum(mat, 0.0)  # only positive damage is meaningful
    ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows if show_y else [])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "black" if mat[i, j] < vmax * 0.5 else "white"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--neurons", required=True, choices=("sln", "mln"))
    args = p.parse_args()

    result_dir = RESULTS_ROOT / model_short_name(args.model)
    panels = []
    for corpus, label in CORPORA:
        csv_path = result_dir / f"selectivity_{args.neurons}_{corpus}.csv"
        if csv_path.is_file():
            panels.append((label,) + _load(csv_path))
        else:
            print(f"missing {csv_path}, skipping")
    if not panels:
        raise SystemExit("no selectivity CSVs found")

    vmax = max(float(np.nanmax(np.maximum(m, 0.0))) for _, m, _, _, _ in panels)
    n_rows = len(panels[0][2])
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(4.0 * len(panels), 0.55 * n_rows + 1.2),
                             squeeze=False)
    for c, (label, mat, rows, cols, n_neurons) in enumerate(panels):
        ax = axes[0, c]
        _draw(ax, mat, rows, cols, vmax, show_y=(c == 0))
        ax.set_title(f"{label} (|N| = {n_neurons})", fontweight="bold")
        ax.set_xlabel("evaluation language")
        if c == 0:
            ax.set_ylabel("target" if args.neurons == "sln" else "target language set")

    out_pdf = result_dir / f"heatmap_{args.neurons}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out_pdf} (+.png)")


if __name__ == "__main__":
    main()
