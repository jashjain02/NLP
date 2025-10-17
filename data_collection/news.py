from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote_plus, urlparse, urlunparse
import json
import urllib.request

import pandas as pd
import pytz
from dateutil import parser as dateparser

import feedparser
from newspaper import Article
from dotenv import load_dotenv
import requests

try:
    from newsapi import NewsApiClient
except Exception:  # pragma: no cover - newsapi optional in tests
    NewsApiClient = None  # type: ignore


# Load environment variables from a .env file if present
load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _canonicalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        # Drop fragments and query for dedupe; keep scheme+netloc+path
        clean = parsed._replace(query="", fragment="")
        # Normalize scheme and netloc to lowercase
        clean = clean._replace(scheme=clean.scheme.lower(), netloc=clean.netloc.lower())
        return urlunparse(clean)
    except Exception:
        return url


def _to_utc(dt_like) -> Optional[datetime]:
    if dt_like is None:
        return None
    try:
        if isinstance(dt_like, datetime):
            dt = dt_like
        else:
            dt = dateparser.parse(str(dt_like))
        if dt.tzinfo is None:
            # Assume UTC if naive
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


# -------------------- NewsAPI --------------------

def fetch_news_newsapi(api_key: Optional[str], query: str, from_date: str, to_date: str, page_size: int = 100, max_results: int = 100) -> pd.DataFrame:
    """Fetch news via NewsAPI, handling pagination and politeness.

    Returns DataFrame with: title, description, content, url, published (tz-aware UTC).
    Saves parquet to data/raw/news_newsapi_{query}.parquet and a JSON alongside.
    """
    api_key = api_key or os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not provided. Set .env or pass api_key explicitly.")
    if NewsApiClient is None:
        raise RuntimeError("newsapi-python is not installed. Add it to requirements and install.")

    client = NewsApiClient(api_key=api_key)

    all_articles = []
    page = 1
    max_pages = 10  # safety cap
    max_results = max(1, int(max_results))

    while page <= max_pages and len(all_articles) < max_results:
        effective_page_size = min(page_size, max_results - len(all_articles))
        resp = client.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language="en",
            sort_by="publishedAt",
            page_size=effective_page_size,
            page=page,
        )
        status = resp.get("status")
        if status != "ok":
            # Break gracefully on max-results errors for dev plan
            if isinstance(resp, dict) and resp.get("code") == "maximumResultsReached":
                break
            raise RuntimeError(f"NewsAPI error: {resp}")
        articles = resp.get("articles", [])
        if not articles:
            break
        for a in articles:
            all_articles.append({
                "title": a.get("title"),
                "description": a.get("description"),
                "content": a.get("content"),
                "url": a.get("url"),
                "published": _to_utc(a.get("publishedAt")),
                "source": (a.get("source") or {}).get("name"),
            })
        # Politeness: small delay to avoid rate limit bursts
        time.sleep(0.5)
        page += 1
        total = resp.get("totalResults", 0)
        if page > (total // effective_page_size + 1):
            break

    df = pd.DataFrame(all_articles[:max_results])
    # Deduplicate by canonical URL
    if not df.empty:
        df["url_canon"] = df["url"].map(lambda u: _canonicalize_url(u or ""))
        df = df.drop_duplicates(subset=["url_canon"]).drop(columns=["url_canon"])  

    # Save outputs
    out_path = CACHE_DIR / f"news_newsapi_{quote_plus(query)}.parquet"
    try:
        df.to_parquet(out_path, index=False)
        # Save raw records JSON as simple line-delimited
        json_path = out_path.with_suffix(".jsonl")
        with open(json_path, "w", encoding="utf-8") as f:
            for rec in all_articles[:max_results]:
                # Lightweight JSON serialization
                f.write(pd.Series(rec).to_json(date_unit="s", date_format="iso"))
                f.write("\n")
    except Exception:
        pass

    return df


# -------------------- MarketAux --------------------

def fetch_news_marketaux(api_key: Optional[str], query: str, from_date: str, to_date: str, max_results: int = 200, symbols: Optional[list[str]] = None) -> pd.DataFrame:
    """Fetch finance news from MarketAux /v1/news/all endpoint.

    Requires API key. Returns DataFrame with: title, description, url, published, source.
    """
    key = api_key or os.getenv("MARKETAUX_API_KEY") or os.getenv("NEWSAPI_KEY")
    if not key:
        raise RuntimeError("MARKETAUX_API_KEY not provided. Set .env or pass api_key explicitly.")

    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "api_token": key,
        "symbols": ",".join(symbols) if symbols else None,
        "filter_entities": "true",
        "language": "en",
        "limit": min(100, max_results),
        "published_after": from_date,
        "published_before": to_date,
        "query": query,
    }

    records = []
    page = 1
    fetched = 0
    while fetched < max_results:
        params["page"] = page
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"MarketAux error: {exc}")

        items = data.get("data", [])
        if not items:
            break
        for it in items:
            records.append({
                "title": it.get("title"),
                "description": it.get("description"),
                "url": it.get("url"),
                "published": _to_utc(it.get("published_at")),
                "source": (it.get("source") or {}).get("name") if isinstance(it.get("source"), dict) else it.get("source"),
            })
        fetched += len(items)
        page += 1
        if len(items) < params["limit"]:
            break

    df = pd.DataFrame(records[:max_results])
    if not df.empty:
        df["url_canon"] = df["url"].map(lambda u: _canonicalize_url(u or ""))
        df = df.drop_duplicates(subset=["url_canon"]).drop(columns=["url_canon"])  

    out_path = CACHE_DIR / f"news_marketaux_{quote_plus(query)}.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        pass
    return df


# -------------------- GDELT + Newspaper3k --------------------

def _gdelt_doc_api(query: str, start_dt: datetime, end_dt: datetime, maxrecords: int = 250) -> list[dict]:
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    start = start_dt.strftime("%Y%m%d%H%M%S")
    end = end_dt.strftime("%Y%m%d%H%M%S")
    params = f"query={quote_plus(query)}&mode=ArtList&format=json&startdatetime={start}&enddatetime={end}&maxrecords={maxrecords}"
    url = f"{base}?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return data.get("articles", [])


def fetch_news_gdelt_and_scrape(query: str, from_date: str, to_date: str, max_articles: int = 1000) -> pd.DataFrame:
    """Fetch historical articles from GDELT Documents API and scrape full text.

    Returns DataFrame with: title, text, url, published (tz-aware UTC), source.
    Saves parquet to data/raw/news_gdelt_{query}.parquet.
    """
    start_dt = dateparser.parse(from_date).replace(tzinfo=timezone.utc)
    end_dt = dateparser.parse(to_date).replace(tzinfo=timezone.utc) + timedelta(hours=23, minutes=59, seconds=59)

    records = []
    chunk = timedelta(days=30)
    cur_start = start_dt
    while cur_start <= end_dt and len(records) < max_articles:
        cur_end = min(cur_start + chunk, end_dt)
        try:
            arts = _gdelt_doc_api(query, cur_start, cur_end, maxrecords=min(250, max_articles - len(records)))
        except Exception:
            arts = []
        for a in arts:
            url = a.get("url")
            if not url:
                continue
            title = a.get("title")
            ts = _to_utc(a.get("seendate") or a.get("published"))
            source = a.get("sourceCommonName") or a.get("domain")
            records.append({
                "title": title,
                "url": url,
                "published": ts,
                "source": source,
            })
        cur_start = cur_end + timedelta(seconds=1)
        time.sleep(0.2)

    df = pd.DataFrame(records)

    # Deduplicate by canonical URL
    if not df.empty:
        df["url_canon"] = df["url"].map(lambda u: _canonicalize_url(u or ""))
        df = df.drop_duplicates(subset=["url_canon"]).drop(columns=["url_canon"])  

    # Scrape text
    texts = []
    for url in df.get("url", []):
        text = None
        try:
            art = Article(url)
            art.download()
            art.parse()
            text = art.text or None
        except Exception:
            text = None
        texts.append(text)
    if len(texts) == len(df):
        df["text"] = texts

    out_path = CACHE_DIR / f"news_gdelt_{quote_plus(query)}.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        pass

    return df


# -------------------- RSS + Newspaper3k --------------------

def _google_news_rss_url(query: str) -> str:
    q = quote_plus(query)
    # Top stories matching the query in English; use when param to avoid duplicates
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def fetch_news_rss_and_scrape(query: str, max_articles: int = 300) -> pd.DataFrame:
    """Fetch news via Google News RSS, then scrape article text with newspaper3k.

    Returns DataFrame with: title, text, url, published (tz-aware UTC), source.
    Saves parquet to data/raw/news_rss_{query}.parquet.
    """
    feed_url = _google_news_rss_url(query)
    parsed = feedparser.parse(feed_url)

    records = []
    seen_urls = set()

    for entry in parsed.entries:
        url = entry.get("link") or entry.get("id")
        if not url:
            continue
        canon = _canonicalize_url(url)
        if canon in seen_urls:
            continue
        seen_urls.add(canon)

        title = entry.get("title")
        published = entry.get("published") or entry.get("updated")
        published_dt = _to_utc(published)
        source = None
        if entry.get("source") and isinstance(entry.get("source"), dict):
            source = entry["source"].get("title")

        text = None
        try:
            art = Article(url)
            art.download()
            art.parse()
            text = art.text or None
        except Exception:
            text = None

        records.append({
            "title": title,
            "text": text,
            "url": url,
            "published": published_dt,
            "source": source,
        })

        if len(records) >= max_articles:
            break

    df = pd.DataFrame(records)

    # Deduplicate by canonical URL
    if not df.empty:
        df["url_canon"] = df["url"].map(lambda u: _canonicalize_url(u or ""))
        df = df.drop_duplicates(subset=["url_canon"]).drop(columns=["url_canon"])  

    out_path = CACHE_DIR / f"news_rss_{quote_plus(query)}.parquet"
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        pass

    return df


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="News collection via NewsAPI, RSS+scrape, or GDELT+scrape")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_a = sub.add_parser("newsapi", help="Fetch via NewsAPI")
    p_a.add_argument("--query", required=True)
    p_a.add_argument("--from_date", required=True)
    p_a.add_argument("--to_date", required=True)
    p_a.add_argument("--page_size", type=int, default=100)
    p_a.add_argument("--max_results", type=int, default=100)
    p_a.add_argument("--api_key", default=None)

    p_b = sub.add_parser("rss", help="Fetch via Google News RSS + newspaper3k")
    p_b.add_argument("--query", required=True)
    p_b.add_argument("--max_articles", type=int, default=300)

    p_c = sub.add_parser("gdelt", help="Fetch via GDELT + newspaper3k")
    p_c.add_argument("--query", required=True)
    p_c.add_argument("--from_date", required=True)
    p_c.add_argument("--to_date", required=True)
    p_c.add_argument("--max_articles", type=int, default=1000)

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        if args.cmd == "newsapi":
            df = fetch_news_newsapi(args.api_key, args.query, args.from_date, args.to_date, args.page_size, args.max_results)
        elif args.cmd == "gdelt":
            df = fetch_news_gdelt_and_scrape(args.query, args.from_date, args.to_date, args.max_articles)
        else:
            df = fetch_news_rss_and_scrape(args.query, args.max_articles)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if "NEWSAPI_KEY" in str(exc):
            print("Hint: set NEWSAPI_KEY in your .env or pass --api_key.", file=sys.stderr)
        return 1

    print(df.head().to_string(index=False))
    print(f"\nRows: {len(df)}  Columns: {list(df.columns)}")
    return 0


def fetch_news_moneycontrol_and_scrape(query: str, from_date: str, to_date: str, max_articles: int = 1000) -> pd.DataFrame:
    """Fetch news from Indian financial RSS feeds and scrape full articles.
    Includes MoneyControl, LiveMint, Business Standard, Economic Times, Financial Express.
    Provides free access to Indian financial news with good historical coverage.
    """
    import requests
    from bs4 import BeautifulSoup
    
    print(f"Fetching MoneyControl news for: {query}")
    
    # Indian financial news RSS feeds
    rss_urls = [
        "https://www.moneycontrol.com/rss/business.xml",
        "https://www.moneycontrol.com/rss/marketnews.xml", 
        "https://www.moneycontrol.com/rss/stockmarket.xml",
        "https://www.moneycontrol.com/rss/companynews.xml",
        "https://www.livemint.com/rss/markets",
        "https://www.business-standard.com/rss/markets-106.rss",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.financialexpress.com/market/rss/"
    ]
    
    all_articles = []
    
    for rss_url in rss_urls:
        try:
            print(f"Fetching from: {rss_url}")
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:max_articles//len(rss_urls)]:
                # Check if article is relevant to our query
                title = entry.get('title', '').lower()
                summary = entry.get('summary', '').lower()
                query_terms = [term.strip().lower() for term in query.split(' OR ')]
                
                if any(term in title or term in summary for term in query_terms):
                    article_data = {
                        'title': entry.get('title', ''),
                        'url': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', ''),
                        'source': 'MoneyControl'
                    }
                    all_articles.append(article_data)
                    
        except Exception as e:
            print(f"Error fetching from {rss_url}: {e}")
            continue
    
    print(f"Found {len(all_articles)} relevant articles from MoneyControl RSS")
    
    # Now scrape full article content
    scraped_articles = []
    for i, article in enumerate(all_articles[:max_articles]):
        try:
            print(f"Scraping article {i+1}/{min(len(all_articles), max_articles)}: {article['title'][:50]}...")
            
            # Use newspaper3k to scrape full content
            article_obj = Article(article['url'])
            article_obj.download()
            article_obj.parse()
            
            if article_obj.text and len(article_obj.text) > 100:  # Only keep substantial articles
                scraped_articles.append({
                    'title': article['title'],
                    'text': article_obj.text,
                    'url': article['url'],
                    'published': article['published'],
                    'source': 'MoneyControl'
                })
            
            time.sleep(0.5)  # Be respectful to the server
            
        except Exception as e:
            print(f"Error scraping article: {e}")
            continue
    
    print(f"Successfully scraped {len(scraped_articles)} articles")
    
    if not scraped_articles:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(scraped_articles)
    
    # Parse dates
    df['published'] = pd.to_datetime(df['published'], errors='coerce')
    df = df.dropna(subset=['published'])
    
    # Filter by date range
    from_date_dt = pd.to_datetime(from_date)
    to_date_dt = pd.to_datetime(to_date)
    df = df[(df['published'] >= from_date_dt) & (df['published'] <= to_date_dt)]
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['url'])
    
    # Save to parquet
    output_path = Path("data/raw") / f"news_moneycontrol_{query.replace(' ', '_')}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} articles to {output_path}")
    
    return df


if __name__ == "__main__":
    raise SystemExit(main())
