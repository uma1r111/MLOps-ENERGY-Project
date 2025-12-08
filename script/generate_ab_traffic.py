"""
Generate A/B test traffic with user feedback
"""
import requests
import time
import random
from typing import List, Dict

API_URL = "http://localhost:8000"

QUESTIONS = [
    "How can households reduce energy consumption?",
    "What are the benefits of solar panels?",
    "How does insulation affect energy efficiency?",
    "What are smart home energy solutions?",
    "How do LED bulbs save energy?",
    "What is renewable energy?",
    "How can I reduce my electricity bill?",
    "What are energy-efficient appliances?",
    "How does weather affect energy use?",
    "What is net zero energy?",
    "How do heat pumps work?",
    "What are the best energy saving tips?",
    "How to conduct an energy audit?",
    "What is sustainable energy?",
    "How can businesses reduce energy costs?",
]

def send_query(question: str, user_id: str = None) -> Dict:
    """Send query to API"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "user_id": user_id},
            headers={"X-User-Id": user_id} if user_id else {},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def submit_feedback(query: str, variant_id: str, score: float):
    """Submit user satisfaction feedback"""
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "query": query,
                "variant_id": variant_id,
                "satisfaction_score": score
            }
        )
        return response.status_code == 200
    except:
        return False

def simulate_user_satisfaction(variant_id: str, latency: float, answer_length: int) -> float:
    """
    Simulate user satisfaction based on variant characteristics.
    
    Different variants have different satisfaction patterns:
    - Control: Balanced, moderate satisfaction
    - Concise: High satisfaction for quick answers
    - Detailed: High satisfaction for comprehensive answers
    - Conversational: Very high satisfaction for friendly tone
    - Technical: Variable based on user type
    """
    base_scores = {
        'control': 3.5,
        'concise': 4.0,        # Users like quick answers
        'detailed': 3.8,       # Good for learning
        'conversational': 4.2, # Friendly tone appreciated
        'technical': 3.3       # Too technical for most users
    }
    
    base = base_scores.get(variant_id, 3.5)
    
    # Latency penalty
    if latency > 2:
        base -= 0.5
    elif latency < 1:
        base += 0.2
    
    # Length preference (simulate different user types)
    user_preference = random.choice(['quick', 'detailed', 'balanced'])
    
    if user_preference == 'quick':
        if answer_length < 200:
            base += 0.3
        elif answer_length > 500:
            base -= 0.4
    elif user_preference == 'detailed':
        if answer_length > 400:
            base += 0.4
        elif answer_length < 100:
            base -= 0.3
    
    # Add some randomness
    base += random.uniform(-0.3, 0.3)
    
    # Clamp to 0-5
    return max(0, min(5, base))

def main():
    print("="*60)
    print("A/B Testing Traffic Generator")
    print("="*60)
    
    # Check API
    try:
        health = requests.get(f"{API_URL}/health").json()
        print(f"✓ API Status: {health['status']}")
        print(f"✓ Variants loaded: {health['variants_loaded']}")
    except:
        print("❌ API not available")
        return
    
    print("\n🚀 Generating A/B test traffic with simulated users...")
    print("This will:")
    print("  - Send queries from different 'users'")
    print("  - Track which variant each user gets")
    print("  - Simulate user satisfaction feedback")
    print()
    
    num_users = 20
    queries_per_user = random.randint(2, 4)
    
    results_summary = {
        'total_queries': 0,
        'by_variant': {},
        'avg_satisfaction': {}
    }
    
    for user_num in range(1, num_users + 1):
        user_id = f"user_{user_num}"
        print(f"\n👤 Simulating User {user_num}/{num_users} ({user_id})")
        
        for query_num in range(queries_per_user):
            question = random.choice(QUESTIONS)
            print(f"  [{query_num+1}/{queries_per_user}] {question[:45]}...")
            
            result = send_query(question, user_id)
            
            if result:
                variant_id = result['variant_id']
                variant_name = result['variant_name']
                latency = result['latency']
                answer_length = len(result['answer'])
                
                print(f"    ✓ Variant: {variant_name}")
                print(f"    ✓ Latency: {latency:.2f}s")
                
                # Track results
                results_summary['total_queries'] += 1
                if variant_id not in results_summary['by_variant']:
                    results_summary['by_variant'][variant_id] = 0
                    results_summary['avg_satisfaction'][variant_id] = []
                results_summary['by_variant'][variant_id] += 1
                
                # Simulate user feedback
                satisfaction = simulate_user_satisfaction(
                    variant_id, latency, answer_length
                )
                
                if submit_feedback(question, variant_id, satisfaction):
                    print(f"    ⭐ Satisfaction: {satisfaction:.1f}/5.0")
                    results_summary['avg_satisfaction'][variant_id].append(satisfaction)
            
            time.sleep(random.uniform(1, 2))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 A/B Test Summary")
    print("="*60)
    print(f"Total Queries: {results_summary['total_queries']}")
    print(f"Unique Users: {num_users}")
    print()
    print("Distribution by Variant:")
    for variant_id, count in results_summary['by_variant'].items():
        percentage = (count / results_summary['total_queries']) * 100
        avg_sat = sum(results_summary['avg_satisfaction'][variant_id]) / len(results_summary['avg_satisfaction'][variant_id])
        print(f"  {variant_id:15} {count:3} queries ({percentage:5.1f}%) - Avg Satisfaction: {avg_sat:.2f}/5.0")
    
    print("\n" + "="*60)
    print("✅ Traffic generation complete!")
    print("="*60)
    print("\nView results:")
    print("  🔹 A/B Stats:  http://localhost:8000/ab-stats")
    print("  🔹 Variants:   http://localhost:8000/variants")
    print("  🔹 Grafana:    http://localhost:3000")
    print("="*60)

if __name__ == "__main__":
    main()