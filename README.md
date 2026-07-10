# Distribution-aware Language Neuron Identification

Code for *Distribution-aware Language Neuron Identification in Multilingual Large Language Models*.

DLN (Distribution-aware Language Neuron) selection identifies language neurons in GLU feed-forward blocks from the **full per-language activation distributions**: each neuron's pairwise overlap-coefficient matrix over languages is bipartitioned by single-linkage clustering (K = 2), and the neuron is selected when its maximum between-cluster overlap falls below a percentile threshold τ. Neurons with a singleton cluster are **single-language neurons (SLNs)**; larger clusters are **multi-language neurons (MLNs)**.

## Setup

```bash
uv sync            # or: pip install -e .
```

Models used in the paper: `meta-llama/Llama-3.1-8B`, `HuggingFaceTB/SmolLM3-3B-Base`.
Languages: en, zh, fr, es, vi, id, ja (FLORES codes).

## Pipeline

All caches are written under `src/cache/`, results under `results/<model>/`.

### 1. Build activation caches

Per-(layer, neuron, language) activation histograms (400 inner bins on a per-neuron adaptive range), from 100k Wikipedia calibration tokens per language:

```bash
python -m src.data.build_hist_cache --model meta-llama/Llama-3.1-8B
```

Per-(layer, neuron, language) mean activations (used as the mean-patch replacement values), from 1M tokens per language:

```bash
python -m src.data.build_activation_means --model meta-llama/Llama-3.1-8B
```

### 2. Select language neurons

Overlap matrix → K=2 single-linkage bipartition → threshold at the bottom-P percentile (paper default P = 1%):

```bash
python -m src.method.select_neurons --model meta-llama/Llama-3.1-8B --pct 1.0
```

Writes `results/<model>/populations.json` with each selected neuron's language cluster.

### 3. Selectivity experiment (main result)

Mean-patch each target's neurons to the mean activation over the non-target languages, then measure the ΔNLL on every evaluation language:

```bash
for corpus in flores_devtest wiki_test; do
    python -m src.eval.run_selectivity --model meta-llama/Llama-3.1-8B --neurons sln --corpus $corpus
    python -m src.eval.run_selectivity --model meta-llama/Llama-3.1-8B --neurons mln --corpus $corpus
done
```

SLN runs one intervention per target language; MLN runs one per cluster (default: the 4 most-populated clusters; override with e.g. `--groups zh,ja fr,es id,vi zh,ja,vi`).

### 4. Heatmap figures

```bash
python -m src.figures.selectivity_heatmap --model meta-llama/Llama-3.1-8B --neurons sln
python -m src.figures.selectivity_heatmap --model meta-llama/Llama-3.1-8B --neurons mln
```

Writes `results/<model>/heatmap_{sln,mln}.pdf`. A diagonal-dominant heatmap means interventions damage the target language while sparing the others.

## Layout

```
src/
├── data/        model/corpus loading, gated-MLP activation capture, cache builders
├── method/      overlap coefficients + clustering-based neuron selection
├── eval/        mean-patch intervention, perplexity, selectivity experiment
└── figures/     ΔNLL heatmap rendering
```
