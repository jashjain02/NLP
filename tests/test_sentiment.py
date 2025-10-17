from unittest import mock

import pandas as pd

from nlp.sentiment import score_texts_finbert, score_vader


class DummyPipeline:
    def __init__(self):
        class Cfg:
            id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        class Model:
            config = Cfg()
        self.model = Model()
    def __call__(self, texts, **kwargs):
        # Alternate labels deterministically
        out = []
        labs = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        for i, t in enumerate(texts):
            lab = labs[i % 3]
            score = 0.7 if lab != 'NEUTRAL' else 0.5
            out.append({"label": lab, "score": score})
        return out


def test_finbert_label_mapping_and_scores():
    pipe = DummyPipeline()
    texts = ["bad", "meh", "good"]
    df = score_texts_finbert(texts, pipe, batch_size=2, cache_source=None)
    assert list(df["label"]) == ["negative", "neutral", "positive"]
    assert all(-1.0 <= s <= 1.0 for s in df["score"])  # signed scores
    assert df.iloc[0]["score"] < 0 and df.iloc[2]["score"] > 0 and df.iloc[1]["score"] == 0


def test_vader_scores_range():
    texts = ["Terrible loss", "It is ok", "Amazing profit"]
    df = score_vader(texts)
    assert set(df.columns) == {"text","label","score"}
    assert all(-1.0 <= s <= 1.0 for s in df["score"])
