"""
Quick RAG system test script - FastEmbed + Google Gemini
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}\n")

def test_query(question):
    """Test query endpoint"""
    print(f"🔍 Testing query: '{question}'")
    response = requests.post(
        f"{BASE_URL}/query",
        json={"question": question, "top_k": 3}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Success!")
        print(f"\n🤖 LLM: {data.get('model', 'unknown')}")
        print(f"📊 Embeddings: {data.get('embedding_model', 'unknown')}")
        print(f"\n📝 Answer:\n{data['answer']}\n")
        print(f"📚 Sources ({len(data['sources'])}):")
        for i, source in enumerate(data['sources'], 1):
            print(f"  {i}. {source['source']}")
        print(f"\n⏱️  Latency: {data['latency']}s")
        print(f"🔢 Tokens used: {data['tokens_used']}\n")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}\n")

def main():
    print("=" * 60)
    print("RAG System Test - FastEmbed + Google Gemini")
    print("=" * 60 + "\n")
    
    # Test health
    test_health()
    
    # Test queries
    test_queries = [
        "How can households reduce energy consumption cost?",
        "How do wind turbines work?",
        "What are the benefits of renewable energy?",
        "Why do energy bills increase during summer and winter?",
        "Does keeping a laptop plugged in all the time increase the bill?",
        "What are the top 10 ways to reduce household electricity consumption?",
        "How can I reduce AC usage without feeling hot?",
        "Does unplugging appliances reduce electricity consumption?"
        "How can I lower my energy bill without buying new appliances?"
    ]
    
    for query in test_queries:
        test_query(query)
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()