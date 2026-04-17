import os
import json
import time
import re
import numpy as np
import feedparser
import requests
import pytz
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ==================== NEWSAPI QUOTA TRACKER ====================
QUOTA_FILE = "newsapi_quota.json"

# Only active, English-language business/finance feeds
ENGLISH_GLOBAL_FEEDS = {
    'US_CNBC':         'https://www.cnbc.com/id/100003114/device/rss/rss.html',
    'US_Yahoo_Fin':    'https://finance.yahoo.com/news/rssindex',
    'US_Investing':    'https://www.investing.com/rss/news.rss',
    'US_MarketWatch':  'https://feeds.marketwatch.com/marketwatch/topstories/',
    'US_Coindesk':     'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'UK_Guardian_Biz': 'https://www.theguardian.com/business/rss',
    'EU_DW_English':   'https://rss.dw.com/rdf/rss-en-bus',
    'AU_ABC_Biz':      'https://www.abc.net.au/news/feed/51892/rss.xml',
    'INTL_AlJazeera':  'https://www.aljazeera.com/xml/rss/all.xml',
    'AE_ArabianBiz':   'https://www.arabianbusiness.com/feed',
    'US_Reuters':      'https://feeds.reuters.com/reuters/businessNews',
    'US_Bloomberg':    'https://feeds.bloomberg.com/markets/news.rss',
    'US_WSJ':          'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    'CA_CBC':          'https://www.cbc.ca/webfeed/rss/rss-business',
    'IN_MoneyControl': 'https://www.moneycontrol.com/rss/latestnews.xml',
    'IN_EconTimes':    'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
    'AU_ABC':          'https://www.abc.net.au/news/feed/51892/rss.xml',
    'AU_SMH':          'https://www.smh.com.au/rss/business.xml',

    # === NEW RELIABLE ADDITIONS ===
    'US_SeekingAlpha': 'https://seekingalpha.com/feed.xml',                    # Excellent for stock analysis & ideas
    'US_Nasdaq':       'https://www.nasdaq.com/feed/rssoutbound?category=Markets',  # Market news
    'US_Forbes_Biz':   'https://www.forbes.com/business/feed/',               # High quality business
    'US_NYT_Business': 'https://rss.nytimes.com/services/xml/rss/nyt/Business.xml',
    'US_Economist':    'https://www.economist.com/latest/rss.xml',            # Premium global view
    'UK_FT_Markets':   'https://www.ft.com/markets?format=rss',               # Financial Times Markets
    'UK_FT_Global':    'https://www.ft.com/global-economy?format=rss',
    'US_Benzinga':     'https://www.benzinga.com/feed',                       # Fast-moving market news
    'US_Barrons':      'https://www.barrons.com/feed',                        # High-quality investing
    'INTL_Reuters_Biz':'https://www.reuters.com/business/feed/',              # More general Reuters business
    'US_Investopedia': 'https://www.investopedia.com/feed',                   # Educational + news
    'US_MotleyFool':   'https://www.fool.com/feed',                           # Popular stock picks
    'US_FinancialTimes':'https://www.ft.com/rss/home',                        # FT main feed
}

_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}


# ==================== NEWSAPI QUOTA TRACKER ====================
def load_quota():
    if os.path.exists(QUOTA_FILE):
        try:
            with open(QUOTA_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "requests_24h": 0,
        "last_reset_24h": datetime.now().isoformat(),
        "requests_12h": 0,
        "last_reset_12h": datetime.now().isoformat()
    }

def save_quota(quota):
    with open(QUOTA_FILE, "w") as f:
        json.dump(quota, f)

def can_use_newsapi():
    quota = load_quota()
    now = datetime.now()

    if now - datetime.fromisoformat(quota["last_reset_24h"]) > timedelta(hours=24):
        quota["requests_24h"] = 0
        quota["last_reset_24h"] = now.isoformat()

    if now - datetime.fromisoformat(quota["last_reset_12h"]) > timedelta(hours=12):
        quota["requests_12h"] = 0
        quota["last_reset_12h"] = now.isoformat()

    if quota["requests_24h"] >= 100 or quota["requests_12h"] >= 50:
        print("   ⚠️  NewsAPI quota reached → skipping to RSS")
        return False

    quota["requests_24h"] += 1
    quota["requests_12h"] += 1
    save_quota(quota)
    return True
# ==================== FETCHING LOGIC ====================

def clean_text(text: str) -> str:
    """Removes HTML tags, URLs, and cleans up whitespace."""
    if not text: return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Clean whitespace
    return " ".join(text.split())

# ==================== THE WORKERS ====================

def fetch_all_rss_sources(ticker: str, target_dt: datetime) -> list:
    """
    Step 7: Robust RSS Fetcher. 
    Matches both Ticker (TSLA) and Company Name (Tesla) to avoid missing news.
    """
    headlines = []
    start_window = target_dt - timedelta(days=2)
    end_window = target_dt + timedelta(days=1)
    
    # --- NEW: TICKER TO NAME MAPPING ---
    # This ensures that even if a headline doesn't say "TSLA", we find it.
    name_map = {
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia",
        "TSLA": "Tesla", "GOOGL": "Google", "AMZN": "Amazon",
        "META": "Facebook", "JPM": "JPMorgan", "SPY": "S&P 500", "QQQ": "Nasdaq"
    }
    company_name = name_map.get(ticker, ticker)
    
    # Improved Regex: Matches "TSLA", "tsla", or "Tesla" as whole words
    pattern = rf"\b({ticker}|{ticker.lower()}|{company_name})\b"
    # -----------------------------------

    for name, url in ENGLISH_GLOBAL_FEEDS.items():
        try:
            # 1. Prepare URL (Inject ticker if it's the Yahoo Ticker feed)
            current_url = url.format(ticker=ticker) if '{ticker}' in url else url
            
            # Use a slightly longer timeout for reliability
            resp = requests.get(current_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if resp.status_code != 200: continue
            
            feed = feedparser.parse(resp.content)
            
            for entry in feed.entries:
                pub = entry.get("published_parsed")
                if not pub: continue
                pub_dt = datetime(*pub[:6], tzinfo=pytz.utc)

                if start_window <= pub_dt <= end_window:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    combined = f"{title} {summary}"

                    # 2. THE SMART MATCH
                    # If it's the Yahoo Ticker feed, we take it 100%
                    # If it's a general feed (CNBC/Guardian), we check the pattern
                    if '{ticker}' in url or re.search(pattern, combined, re.IGNORECASE):
                        # Clean and add
                        cleaned = clean_text(title)
                        if cleaned: headlines.append(cleaned)
            
        except Exception:
            continue 

    # Deduplicate and return
    unique_h = list(set(headlines))
    if unique_h:
        print(f"   ✅ RSS Scan: Found {len(unique_h)} unique headlines for {ticker}")
    return unique_h



def get_sentiment_hf_api(headlines: list) -> float:
    import requests
    API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    scores = []
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    for headline in headlines[:6]:
        try:
            resp = requests.post(API_URL, headers=headers, json={"inputs": headline}, timeout=12)
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and result:
                    top = max(result[0], key=lambda x: x["score"])
                    scores.append(label_map.get(top["label"].lower(), 0.0))
        except:
            continue
    return float(np.mean(scores)) if scores else 0.0


def get_sentiment_local(headlines: list) -> float:
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from transformers import pipeline, logging
    logging.set_verbosity_error()

    pipe = pipeline("text-classification", model="ProsusAI/finbert", truncation=True, max_length=512)
    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = []
    
    for i, h in enumerate(headlines[:6]):
        try:
            result = pipe(h)
            label = result[0]["label"].lower()
            val = scores.append(label_map.get(label, 0.0))
            print(f"[Sub-Score {i+1}] {val:+.2f} | {h[:50]}...") 
            scores.append(val)
        except:
            continue
    return float(np.mean(scores)) if scores else 0.0



def fetch_headlines(ticker: str, date_str: str) -> list:
    """
    Get recent headlines:
    1. Try NewsAPI (last 2-3 days)
    2. Fallback to RSS (last 24-48 hours)
    """
    target_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=pytz.utc)
    now = datetime.now(pytz.utc)

    headlines = []

    # =========================
    # 1. NEWSAPI (LAST 3 DAYS)
    # =========================
    if os.getenv("NEWS_API_KEY") and can_use_newsapi():
        try:
            from newsapi import NewsApiClient
            client = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

            from_dt = (target_dt - timedelta(days=2)).strftime("%Y-%m-%d")
            to_dt   = target_dt.strftime("%Y-%m-%d")

            result = client.get_everything(
                q=f"{ticker} OR {ticker} stock",
                from_param=from_dt,
                to=to_dt,
                language="en",
                sort_by="relevancy",
                page_size=15
            )

            articles = result.get("articles", [])
            newsapi_h = [
                clean_text(a["title"])
                for a in articles
                if a.get("title")
            ]

            if newsapi_h:
                print(f"   ✅ NewsAPI: {len(newsapi_h)} headlines")
                return list(set(newsapi_h))

        except Exception as e:
            print(f"   ⚠️ NewsAPI failed: {e}")

    # =========================
    # 2. RSS FALLBACK (LAST 48H)
    # =========================
    print("   → Using RSS fallback (last 48h)...")

    headlines = fetch_all_rss_sources(ticker, now)

    if not headlines:
        print(f"   ❌ No RSS headlines found for {ticker}")
    else:
        print(f"   ✅ RSS headlines: {len(headlines)}")

    return headlines


def get_sentiment(
    ticker: str,
    date: str,
    use_local: bool = True,
) -> dict:
    print(f"   → Fetching headlines for {ticker} on {date}...")
    headlines = fetch_headlines(ticker, date)

    if not headlines:
        print(f"   ⚠️  No headlines found — sentiment = 0.0 (neutral)")
        return {"score": 0.0, "headlines": [], "label": "neutral"}

    print(f"   → Running FinBERT on {len(headlines)} headlines...")
    score = get_sentiment_local(headlines) if use_local else get_sentiment_hf_api(headlines)

    # Do not strongly amplify weak RSS-derived sentiment scores
    label = "bullish" if score > 0.25 else "bearish" if score < -0.25 else "neutral"

    print(f"   ✅ Sentiment: {score:+.3f} ({label})")
    return {
        "score": score,
        "label": label,
        "headlines": headlines[:12],
    }


if __name__ == "__main__":
    from datetime import date
    TODAY = date.today().strftime("%Y-%m-%d")
    print("=" * 55)
    print("  FINBERT SENTIMENT TEST")
    print("=" * 55)
    result = get_sentiment("AAPL", TODAY, use_local=True)
    print(f"\n📰 Headlines used:")
    for i, h in enumerate(result["headlines"], 1):
        print(f"   [{i}] {h[:100]}")
    print(f"\n🎭 Sentiment Score : {result['score']:+.3f}")
    print(f"🏷️  Label          : {result['label']}")
    print("\n✅ FinBERT sentiment ready")