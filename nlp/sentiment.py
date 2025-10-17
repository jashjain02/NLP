from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from tqdm import tqdm  # optional; if not installed, we fallback to prints

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

# Directories for caching
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


def _device_index() -> int:
    try:
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1


def load_finbert(model_name: str = "ProsusAI/finbert"):
    """Load a FinBERT sentiment pipeline.

    - Auto-detects GPU availability; uses device 0 if CUDA is available, else CPU.
    - Returns a HuggingFace pipeline for text-classification with softmax scores.
    """
    device = _device_index()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tok,
        device=device,
        framework="pt",
        return_all_scores=False,
    )
    return pipe


def _standardize_label(label: str, id2label: Optional[Dict[int, str]] = None) -> str:
    if not label:
        return "neutral"
    lbl = label.strip().lower()
    # Map common label variants
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "label_2": "positive",
        "negative": "negative",
        "neg": "negative",
        "label_0": "negative",
        "neutral": "neutral",
        "label_1": "neutral",
    }
    if lbl in mapping:
        return mapping[lbl]
    # Try id2label mapping like {0:'negative',1:'neutral',2:'positive'}
    if id2label is not None:
        inv = {v.lower(): v.lower() for v in id2label.values()}
        if lbl in inv:
            return inv[lbl]
    return lbl


def _signed_score(label: str, prob: float) -> float:
    lab = _standardize_label(label)
    if lab == "positive":
        return float(prob)
    if lab == "negative":
        return float(-prob)
    return 0.0


def _shorten(text: str, max_length: int = 512) -> str:
    # Heuristic: rely on tokenizer truncation; this function guards extreme long strings
    if text is None:
        return ""
    if len(text) <= max_length * 4:  # rough char-to-token heuristic
        return text
    return text[: max_length * 4]


def _hash_texts(texts: List[str]) -> str:
    m = hashlib.sha256()
    for t in texts:
        m.update((t or "").encode("utf-8", errors="ignore"))
        m.update(b"\0")
    return m.hexdigest()[:12]


def _maybe_load_cache(cache_source: Optional[str], texts: List[str]) -> Optional[pd.DataFrame]:
    if not cache_source:
        return None
    path = PROC_DIR / f"sentiment_{cache_source}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        # Minimal validation: same or superset of texts
        if "text" not in df.columns:
            return None
        # If it contains all requested texts, return in input order
        if set(texts).issubset(set(df["text"].tolist())):
            order = {t: i for i, t in enumerate(texts)}
            df = df[df["text"].isin(texts)].copy()
            df["_ord"] = df["text"].map(order)
            df = df.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
            return df
    except Exception:
        return None
    return None


def _save_cache(cache_source: Optional[str], df: pd.DataFrame) -> None:
    if not cache_source:
        return
    path = PROC_DIR / f"sentiment_{cache_source}.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception:
        pass


def score_texts_finbert(
    texts: List[str],
    model_pipeline,
    batch_size: int = 16,
    cache_source: Optional[str] = None,
) -> pd.DataFrame:
    """Score texts with FinBERT.

    - Returns DataFrame columns: [text, label, score]
    - label mapped to: positive|neutral|negative
    - score in [-1, 1] where sign reflects sentiment polarity
    - Uses simple caching when cache_source is provided
    """
    texts = [t if isinstance(t, str) else ("" if t is None else str(t)) for t in texts]

    cached = _maybe_load_cache(cache_source, texts)
    if cached is not None:
        return cached

    id2label = None
    try:
        id2label = getattr(getattr(model_pipeline, "model", None), "config", None)
        if id2label is not None:
            id2label = getattr(id2label, "id2label", None)
    except Exception:
        id2label = None

    outputs: List[Dict[str, Any]] = []

    total = len(texts)
    rng = range(0, total, max(1, batch_size))
    use_tqdm = False
    try:
        _ = tqdm  # noqa: F841
        use_tqdm = True
    except Exception:
        use_tqdm = False

    iterator = tqdm(rng, desc="Scoring FinBERT", unit="batch") if use_tqdm else rng
    for start in iterator:
        batch = texts[start : start + batch_size]
        short_batch = [_shorten(x) for x in batch]
        preds = model_pipeline(short_batch, truncation=True, max_length=512, batch_size=batch_size)
        if isinstance(preds, dict):
            preds = [preds]
        outputs.extend(preds)

    rows = []
    for text, pred in zip(texts, outputs):
        label = _standardize_label(pred.get("label", ""), id2label=id2label)
        prob = float(pred.get("score", 0.0))
        rows.append({
            "text": text,
            "label": label,
            "score": _signed_score(label, prob),
        })
    df = pd.DataFrame(rows)

    _save_cache(cache_source, df)
    return df


def score_vader(texts: List[str], cache_source: Optional[str] = None) -> pd.DataFrame:
    """Score texts with VADER baseline.

    Returns DataFrame with columns: [text, label, score]
    - label: positive/neutral/negative determined by compound threshold (0.05/-0.05)
    - score: VADER compound in [-1, 1]
    """
    texts = [t if isinstance(t, str) else ("" if t is None else str(t)) for t in texts]

    cached = _maybe_load_cache(cache_source, texts)
    if cached is not None:
        return cached

    sia = SentimentIntensityAnalyzer()
    rows = []
    for t in texts:
        comp = float(sia.polarity_scores(t)["compound"])
        if comp > 0.05:
            lab = "positive"
        elif comp < -0.05:
            lab = "negative"
        else:
            lab = "neutral"
        rows.append({"text": t, "label": lab, "score": comp})

    df = pd.DataFrame(rows)
    _save_cache(cache_source or "vader", df)
    return df
