import os
import sys
import time
import re
from datetime import date

TODAY = date.today().strftime("%Y-%m-%d")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'TradingAgents'))

from dotenv import load_dotenv
load_dotenv()

import importlib.util

def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Apply both patches BEFORE importing TradingAgentsGraph
indicators_mod = load_module("local_indicators", os.path.join(ROOT, "3_market_data", "local_indicators.py"))
indicators_mod.patch_tradingagents()

news_mod = load_module("news_patch", os.path.join(ROOT, "3_market_data", "news_patch.py"))
news_mod.patch_news()

from openai import APIStatusError
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

MODEL_FALLBACK_CHAIN = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]

def run_single_analyst(analyst: str, ticker: str, date: str, model: str) -> dict:
    cfg = {
        **DEFAULT_CONFIG,
        "llm_provider": "groq",
        "deep_think_llm": model,
        "quick_think_llm": model,
        "backend_url": "https://api.groq.com/openai/v1",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "results_dir": "./5_backtesting/results",
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        },
    }
    ta = TradingAgentsGraph(
        selected_analysts=[analyst],
        debug=False,
        config=cfg,
    )
    return ta.propagate(company_name=ticker, trade_date=date)


def run_with_fallback(ticker, date):
    analysts = ["market", "fundamentals", "news"]
    reports = {
        "market_report": "",
        "fundamentals_report": "",
        "news_report": "",
        "final_trade_decision": "",
        "investment_plan": "",
    }

    # Run each analyst independently with fallback
    for analyst in analysts:
        print(f"\n📊 Running {analyst} analyst...")
        success = False

        for model in MODEL_FALLBACK_CHAIN:
            print(f"   🤖 Trying {model}...")
            try:
                state, _ = run_single_analyst(analyst, ticker, date, model)
                report_key = f"{analyst}_report"
                if analyst == "market":
                    report_key = "market_report"
                reports[report_key] = state.get(report_key, "")
                print(f"   ✅ {analyst} done with {model}")
                success = True
                time.sleep(5)  # small buffer between analysts
                break

            except APIStatusError as e:
                err_str = str(e)
                if "tokens per day" in err_str or "TPD" in err_str:
                    print(f"   🔄 Daily limit on {model} → next model")
                    continue
                elif "tokens per minute" in err_str or "TPM" in err_str:
                    wait = parse_wait_time(err_str)
                    print(f"   ⏳ TPM limit. Waiting {wait}s...")
                    time.sleep(wait)
                    try:
                        state, _ = run_single_analyst(analyst, ticker, date, model)
                        report_key = f"{analyst}_report"
                        reports[report_key] = state.get(report_key, "")
                        print(f"   ✅ {analyst} done with {model} after wait")
                        success = True
                        break
                    except:
                        print(f"   🔄 Still failing → next model")
                        continue
                else:
                    print(f"   🔄 Error {e.status_code} → next model")
                    continue
            except Exception as e:
                print(f"   ❌ {type(e).__name__}: {str(e)[:150]} → next model")
                continue

        if not success:
            print(f"   ⚠️  {analyst} analyst failed on all models — using empty report")

    # Final decision run with whatever reports we have
    print(f"\n⚖️  Running final decision...")
    for model in MODEL_FALLBACK_CHAIN:
        try:
            state, signal = run_single_analyst("market", ticker, date, model)
            reports["final_trade_decision"] = state.get("final_trade_decision", "")
            reports["signal"] = signal
            print(f"   ✅ Final decision done with {model}")
            break
        except Exception as e:
            err_str = str(e)
            if "tokens per day" in err_str or "TPD" in err_str:
                continue
            print(f"   ❌ {type(e).__name__}: {str(e)[:150]}")
            continue

    return reports


print("=" * 55)
print("  VIRTUAL TRADING FIRM — FULL PIPELINE TEST")
print("=" * 55)
print(f"  Ticker  : AAPL")
print(f"  Date    : {TODAY}")
print(f"  Mode    : isolated agents + model fallback")
print("=" * 55)

reports = run_with_fallback("AAPL", TODAY)

print("\n" + "=" * 55)
print("  RESULTS")
print("=" * 55)
print(f"\n📊 Market    : {reports.get('market_report','')[:200]}...")
print(f"\n📈 Fundament : {reports.get('fundamentals_report','')[:200]}...")
print(f"\n📰 News      : {reports.get('news_report','')[:200]}...")
print(f"\n⚖️  Decision  : {reports.get('final_trade_decision','')[:200]}...")
print(f"\n🚦 Signal    : {reports.get('signal','N/A')}")
print("\n" + "=" * 55)