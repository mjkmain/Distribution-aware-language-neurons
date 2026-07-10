from __future__ import annotations

from typing import Iterable

from datasets import load_dataset

from src.utils import CACHE_ROOT

LANGS = ["eng_Latn", "cmn_Hans", "fra_Latn", "spa_Latn", "vie_Latn", "ind_Latn", "jpn_Jpan"]
SHORT = {
    "eng_Latn": "en", "cmn_Hans": "zh", "fra_Latn": "fr", "spa_Latn": "es",
    "vie_Latn": "vi", "ind_Latn": "id", "jpn_Jpan": "ja",
}


def load_flores(languages: Iterable[str], split: str = "devtest", n_samples: int | None = None):
    out: dict[str, list[str]] = {}
    for code in languages:
        ds = load_dataset("openlanguagedata/flores_plus", code, split=split)
        texts = list(ds["text"])
        out[code] = texts[:n_samples] if n_samples is not None else texts
    return out


def _wiki_stream(source: str, lang: str):
    w = SHORT[lang]
    if source == "calib":
        return load_dataset("wikimedia/wikipedia", f"20231101.{w}", split="train", streaming=True)
    # held-out dump, disjoint from calib by source and date
    files = f"hf://datasets/graelo/wikipedia/data/20230601/{w}/train-*.parquet"
    return load_dataset("parquet", data_files=files, split="train", streaming=True)


def _wiki_path(source: str, lang: str):
    p = CACHE_ROOT / "wiki"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{source}_{lang}.txt"


def build_wiki_cache(
    languages: Iterable[str],
    *,
    source: str = "calib",
    n_paragraphs: int = 1000,
    min_len: int = 50,
    max_len: int = 500,
    force: bool = False,
) -> None:
    for lang in languages:
        path = _wiki_path(source, lang)
        if path.exists() and not force:
            continue
        collected: list[str] = []
        for record in _wiki_stream(source, lang):
            for line in record.get("text", "").split("\n"):
                line = line.strip()
                if min_len <= len(line) <= max_len:
                    collected.append(line)
            if len(collected) >= n_paragraphs:
                collected = collected[:n_paragraphs]
                break
        path.write_text("\n".join(collected), encoding="utf-8")


def load_wiki(languages: Iterable[str], *, source: str = "calib") -> dict[str, list[str]]:
    languages = list(languages)
    missing = [lg for lg in languages if not _wiki_path(source, lg).exists()]
    if missing:
        build_wiki_cache(missing, source=source)
    return {
        lg: [ln for ln in _wiki_path(source, lg).read_text(encoding="utf-8").splitlines() if ln]
        for lg in languages
    }


def wiki_texts_for_tokens(lang: str, *, source: str, target_tokens: int) -> list[str]:
    # ~100 tokens per paragraph, x3 safety margin
    needed = max(1000, (target_tokens // 100) * 3)
    path = _wiki_path(source, lang)
    have = sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
    if have < needed:
        build_wiki_cache([lang], source=source, n_paragraphs=needed, force=True)
    return load_wiki([lang], source=source)[lang]
