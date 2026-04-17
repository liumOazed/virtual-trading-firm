import os
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from langchain_core.tools import tool
from typing import Annotated


def _get_newsapi_client():
    key = os.getenv("NEWS_API_KEY")
    if not key:
        raise ValueError("NEWS_API_KEY not found in .env")
    return NewsApiClient(api_key=key)


def _format_articles(articles: list, label: str) -> str:
    if not articles:
        return f"No {label} articles found."
    lines = [f"=== {label} ==="]
    for i, a in enumerate(articles[:5], 1):
        title       = a.get("title", "No title")
        source      = a.get("source", {}).get("name", "Unknown")
        published   = a.get("publishedAt", "")[:10]
        description = a.get("description") or "No description"
        lines.append(
            f"\n[{i}] {title}\n"
            f"    Source: {source} | Date: {published}\n"
            f"    {description[:200]}"
        )
    return "\n".join(lines)


@tool
def get_news_newsapi(
    ticker: Annotated[str, "Ticker symbol e.g. AAPL"],
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """Retrieve company-specific news via NewsAPI with yfinance fallback."""
    try:
        client = _get_newsapi_client()
        result = client.get_everything(
            q=ticker,
            from_param=start_date,
            to=end_date,
            language="en",
            sort_by="relevancy",
            page_size=5,
        )
        articles = result.get("articles", [])

        # If NewsAPI returns nothing (historical date) fall back to yfinance
        if not articles:
            print(f"   ⚠️  NewsAPI returned no results for {ticker} — falling back to yfinance news")
            from tradingagents.dataflows.yfinance_news import get_news_yfinance
            return get_news_yfinance(ticker, start_date, end_date)

        return _format_articles(articles, f"News for {ticker} ({start_date} to {end_date})")
    except Exception as e:
        print(f"   ⚠️  NewsAPI error — falling back to yfinance: {e}")
        try:
            from tradingagents.dataflows.yfinance_news import get_news_yfinance
            return get_news_yfinance(ticker, start_date, end_date)
        except Exception as e2:
            return f"No news available: {e2}"


@tool
def get_global_news_newsapi(
    curr_date: Annotated[str, "Current date yyyy-mm-dd"],
    look_back_days: Annotated[int, "Days to look back"] = 7,
    limit: Annotated[int, "Max articles"] = 5,
) -> str:
    """Retrieve global financial news via NewsAPI with yfinance fallback."""
    try:
        client = _get_newsapi_client()
        end   = datetime.strptime(curr_date, "%Y-%m-%d")
        start = end - timedelta(days=look_back_days)
        result = client.get_everything(
            q="stock market OR economy OR Federal Reserve OR inflation OR GDP",
            from_param=start.strftime("%Y-%m-%d"),
            to=end.strftime("%Y-%m-%d"),
            language="en",
            sort_by="relevancy",
            page_size=limit,
        )
        articles = result.get("articles", [])

        if not articles:
            print(f"   ⚠️  NewsAPI no global results — falling back to yfinance")
            from tradingagents.dataflows.yfinance_news import get_global_news_yfinance
            return get_global_news_yfinance(curr_date, look_back_days, limit)

        return _format_articles(articles, f"Global Macro News (last {look_back_days} days)")
    except Exception as e:
        print(f"   ⚠️  NewsAPI global error — falling back to yfinance: {e}")
        try:
            from tradingagents.dataflows.yfinance_news import get_global_news_yfinance
            return get_global_news_yfinance(curr_date, look_back_days, limit)
        except Exception as e2:
            return f"No global news available: {e2}"


def patch_news():
    """
    Patch news tools at every level — module namespace, agent_utils,
    and news_analyst's local reference (captured at import time).
    """
    import tradingagents.agents.utils.agent_utils as agent_utils
    import tradingagents.agents.utils.news_data_tools as news_tools
    import tradingagents.agents.analysts.news_analyst as news_analyst_mod
    import tradingagents.graph.trading_graph as graph_mod

    # Patch agent_utils namespace
    agent_utils.get_news        = get_news_newsapi
    agent_utils.get_global_news = get_global_news_newsapi

    # Patch news_data_tools namespace
    news_tools.get_news         = get_news_newsapi
    news_tools.get_global_news  = get_global_news_newsapi

    # Patch news_analyst module's own imported references
    news_analyst_mod.get_news        = get_news_newsapi
    news_analyst_mod.get_global_news = get_global_news_newsapi

    # Patch the ToolNode inside trading_graph's tool_nodes dict
    # This covers the tool executor that actually runs the tools
    from langgraph.prebuilt import ToolNode
    graph_mod_news_toolnode = ToolNode([get_news_newsapi, get_global_news_newsapi])

    # Store patched toolnode so trading_graph picks it up
    if hasattr(graph_mod, '_patched_news_toolnode'):
        graph_mod._patched_news_toolnode = graph_mod_news_toolnode
    else:
        graph_mod._patched_news_toolnode = graph_mod_news_toolnode
    
    # Override _create_tool_nodes to inject our news tools
    original_create_tool_nodes = graph_mod.TradingAgentsGraph._create_tool_nodes

    def patched_create_tool_nodes(self):
        nodes = original_create_tool_nodes(self)
        nodes["news"] = ToolNode([get_news_newsapi, get_global_news_newsapi])
        nodes["social"] = ToolNode([get_news_newsapi])
        return nodes

    graph_mod.TradingAgentsGraph._create_tool_nodes = patched_create_tool_nodes

    print("✅ NewsAPI patch applied at all levels")