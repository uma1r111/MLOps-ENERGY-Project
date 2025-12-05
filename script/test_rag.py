"""
Test script for RAG API
Tests the FastAPI endpoints with sample queries
"""
import requests
import json
import sys

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

def test_query(question: str):
    """Test query endpoint"""
    print(f"🔍 Testing query: '{question}'")
    
    try:
        payload = {
            "question": question,
            "top_k": 3,
            "include_sources": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"❌ Query failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
        
        result = response.json()
        
        print("✅ Success!")
        print(f"🤖 LLM: {result['model']}")
        print(f"📊 Embeddings: {result['embedding_model']}")
        print(f"⏱️  Latency: {result['latency']}s")
        
        print(f"\n📝 Answer:\n{result['answer']}\n")
        
        if result.get('sources'):
            print(f"📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                # Access dictionary keys instead of object attributes
                content = source.get('content', '')
                source_name = source.get('source', 'unknown')
                score = source.get('retrieval_score')
                
                print(f"\n{i}. {content[:200]}...")
                print(f"   Source: {source_name}")
                if score is not None:
                    print(f"   Score: {score:.3f}")
        
        if result.get('langsmith_trace'):
            print(f"\n📊 {result['langsmith_trace']}")
        
        print("\n" + "="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"❌ Query failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests"""
    print("=" * 60)
    print("RAG System Test - FastEmbed + Google Gemini")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("❌ Health check failed. Is the server running?")
        print("Start it with: python src/app.py")
        sys.exit(1)
    
    # Test queries
    queries = [
        "How can households reduce energy consumption cost?",
        "What are the benefits of solar panels?",
        "What factors affect energy efficiency in buildings?"
    ]
    
    for query in queries:
        if not test_query(query):
            break
    
    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()