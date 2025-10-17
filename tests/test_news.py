import types
from datetime import datetime, timezone
from unittest import mock

import pandas as pd
import pytest

from data_collection.news import fetch_news_rss_and_scrape, fetch_news_newsapi


class DummyArticle:
    def __init__(self, url):
        self.url = url
        self.text = None
    def download(self):
        pass
    def parse(self):
        self.text = f"Parsed content for {self.url}"


def test_fetch_news_rss_and_scrape(monkeypatch):
    fake_feed = types.SimpleNamespace(entries=[
        {"title": "T1", "link": "https://example.com/a?x=1", "published": "2023-01-01T00:00:00Z", "source": {"title": "SrcA"}},
        {"title": "T2", "link": "https://example.com/a?x=2", "published": "2023-01-01T01:00:00Z", "source": {"title": "SrcA"}},
        {"title": "T3", "link": "https://example.com/b", "published": "2023-01-02T00:00:00Z", "source": {"title": "SrcB"}},
    ])

    def fake_parse(url):
        return fake_feed

    monkeypatch.setattr("data_collection.news.feedparser.parse", fake_parse)
    monkeypatch.setattr("data_collection.news.Article", DummyArticle)

    df = fetch_news_rss_and_scrape("OpenAI", max_articles=10)
    assert not df.empty
    # url dedup should keep 2 rows, since two links canonicalize to same path
    assert len(df) == 2
    assert set(["title","text","url","published","source"]).issubset(df.columns)
    assert all(isinstance(ts, datetime) and ts.tzinfo is not None for ts in df["published"].dropna())


def test_fetch_news_newsapi(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self._called = 0
        def get_everything(self, **kwargs):
            self._called += 1
            if self._called == 1:
                return {
                    "status": "ok",
                    "totalResults": 2,
                    "articles": [
                        {"title": "A", "description": "d", "content": "c", "url": "https://ex.com/x", "publishedAt": "2024-01-01T00:00:00Z", "source": {"name": "S"}},
                        {"title": "Adup", "description": "d", "content": "c", "url": "https://ex.com/x?y=1", "publishedAt": "2024-01-01T00:00:00Z", "source": {"name": "S"}},
                    ]
                }
            else:
                return {"status": "ok", "totalResults": 2, "articles": []}

    monkeypatch.setenv("NEWSAPI_KEY", "dummy")
    monkeypatch.setattr("data_collection.news.NewsApiClient", FakeClient)

    df = fetch_news_newsapi(None, "AI", "2024-01-01", "2024-01-02", page_size=100)
    assert not df.empty
    # url dedup should reduce to 1
    assert len(df) == 1
    assert set(["title","description","content","url","published","source"]).issubset(df.columns)
