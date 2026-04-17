import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TradingAgents'))

from dotenv import load_dotenv
from tradingagents.llm_clients.factory import create_llm_client

load_dotenv()

def test_tradingagents_groq():
    print("\n🔧 Testing TradingAgents + Groq...")
    try:
        client = create_llm_client(
            provider="groq",
            model="llama-3.3-70b-versatile",
        )
        llm = client.get_llm()
        response = llm.invoke("Say 'TradingAgents + Groq connected.' only.")
        print(f"✅ TradingAgents + Groq OK → {response.content.strip()}")
        return True
    except Exception as e:
        print(f"❌ Failed → {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("  TRADINGAGENTS GROQ INTEGRATION TEST")
    print("=" * 50)
    ok = test_tradingagents_groq()
    print("=" * 50)
    if ok:
        print("  🚀 TradingAgents is Groq-powered. Ready for Section 2.")
    else:
        print("  ⚠️  Fix the error above before proceeding.")
    print("=" * 50)