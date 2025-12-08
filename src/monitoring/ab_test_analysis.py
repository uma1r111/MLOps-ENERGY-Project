"""
A/B Test Statistical Analysis
Analyzes results and determines statistical significance
"""
import json
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class ABTestAnalyzer:
    """Analyze A/B test results with statistical rigor"""
    
    def __init__(self, results_file: str = "monitoring/ab_test_results.jsonl"):
        self.results_file = results_file
        self.df = self.load_results()
    
    def load_results(self) -> pd.DataFrame:
        """Load results into DataFrame"""
        results = []
        with open(self.results_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        df = pd.DataFrame(results)
        print(f"✓ Loaded {len(df)} test results")
        return df
    
    def summary_stats(self) -> pd.DataFrame:
        """Calculate summary statistics by variant"""
        summary = self.df.groupby('variant_id').agg({
            'latency': ['mean', 'std', 'count'],
            'tokens_input': 'sum',
            'tokens_output': 'sum',
            'cost': ['sum', 'mean'],
            'satisfaction_score': ['mean', 'std', 'count']
        }).round(4)
        
        print("\n" + "="*80)
        print("📊 Summary Statistics by Variant")
        print("="*80)
        print(summary)
        print()
        
        return summary
    
    def t_test_latency(self, variant_a: str, variant_b: str) -> Dict:
        """T-test for latency difference"""
        a_data = self.df[self.df['variant_id'] == variant_a]['latency']
        b_data = self.df[self.df['variant_id'] == variant_b]['latency']
        
        t_stat, p_value = stats.ttest_ind(a_data, b_data)
        
        result = {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'mean_a': a_data.mean(),
            'mean_b': b_data.mean(),
            'diff': a_data.mean() - b_data.mean(),
            'diff_pct': ((a_data.mean() - b_data.mean()) / a_data.mean() * 100),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        return result
    
    def t_test_satisfaction(self, variant_a: str, variant_b: str) -> Dict:
        """T-test for satisfaction difference"""
        a_data = self.df[self.df['variant_id'] == variant_a]['satisfaction_score'].dropna()
        b_data = self.df[self.df['variant_id'] == variant_b]['satisfaction_score'].dropna()
        
        if len(a_data) < 2 or len(b_data) < 2:
            return {
                'error': 'Insufficient data for t-test',
                'variant_a': variant_a,
                'variant_b': variant_b
            }
        
        t_stat, p_value = stats.ttest_ind(a_data, b_data)
        
        result = {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'mean_a': a_data.mean(),
            'mean_b': b_data.mean(),
            'diff': a_data.mean() - b_data.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        return result
    
    def compare_all_variants(self, metric: str = 'latency'):
        """Compare all variants with ANOVA"""
        variants = self.df['variant_id'].unique()
        groups = [self.df[self.df['variant_id'] == v][metric].values for v in variants]
        
        f_stat, p_value = stats.f_oneway(*groups)
        
        print("\n" + "="*80)
        print(f"📈 ANOVA Test - {metric.upper()}")
        print("="*80)
        print(f"Null Hypothesis: All variants have equal mean {metric}")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("✅ SIGNIFICANT: At least one variant is different (p < 0.05)")
        else:
            print("❌ NOT SIGNIFICANT: No significant difference between variants")
        print()
        
        return {'f_statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}
    
    def pairwise_comparison(self, metric: str = 'latency'):
        """Pairwise comparison of all variants"""
        variants = self.df['variant_id'].unique()
        
        print("\n" + "="*80)
        print(f"🔄 Pairwise T-Tests - {metric.upper()}")
        print("="*80)
        
        results = []
        for i, var_a in enumerate(variants):
            for var_b in variants[i+1:]:
                if metric == 'latency':
                    result = self.t_test_latency(var_a, var_b)
                elif metric == 'satisfaction_score':
                    result = self.t_test_satisfaction(var_a, var_b)
                
                if 'error' not in result:
                    sig = "✅ SIGNIFICANT" if result['significant'] else "❌ Not significant"
                    print(f"{var_a:15} vs {var_b:15} | Diff: {result['diff']:+.4f} | p={result['p_value']:.4f} | {sig}")
                    results.append(result)
        
        print()
        return results
    
    def recommend_winner(self) -> Dict:
        """Recommend best variant based on multiple metrics"""
        print("\n" + "="*80)
        print("🏆 WINNER RECOMMENDATION")
        print("="*80)
        
        variants = self.df['variant_id'].unique()
        scores = {}
        
        for variant in variants:
            data = self.df[self.df['variant_id'] == variant]
            
            # Scoring factors (lower is better for latency, higher for satisfaction)
            latency_score = 1 / (data['latency'].mean() + 0.1)
            cost_score = 1 / (data['cost'].mean() + 0.001)
            satisfaction_score = data['satisfaction_score'].mean() if data['satisfaction_score'].notna().any() else 0
            
            # Weighted total (adjust weights as needed)
            total_score = (
                latency_score * 0.30 +      # Speed: 30%
                cost_score * 0.20 +         # Cost: 20%
                satisfaction_score * 0.50   # Satisfaction: 50%
            )
            
            scores[variant] = {
                'latency': data['latency'].mean(),
                'cost': data['cost'].mean(),
                'satisfaction': satisfaction_score,
                'total_score': total_score
            }
        
        # Find winner
        winner = max(scores.items(), key=lambda x: x[1]['total_score'])
        
        print(f"Winner: {winner[0].upper()}")
        print(f"  Total Score: {winner[1]['total_score']:.4f}")
        print(f"  Avg Latency: {winner[1]['latency']:.3f}s")
        print(f"  Avg Cost: ${winner[1]['cost']:.6f}")
        print(f"  Avg Satisfaction: {winner[1]['satisfaction']:.2f}/5.0")
        print()
        
        print("All Variants (ranked):")
        for variant, score_data in sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True):
            print(f"  {variant:15} Score: {score_data['total_score']:.4f} | "
                  f"Latency: {score_data['latency']:.3f}s | "
                  f"Satisfaction: {score_data['satisfaction']:.2f}/5.0")
        
        print("="*80)
        
        return {'winner': winner[0], 'scores': scores}
    
    def plot_distributions(self, save_path: str = "monitoring/ab_test_plots.png"):
        """Create visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Latency distribution
        self.df.boxplot(column='latency', by='variant_id', ax=axes[0, 0])
        axes[0, 0].set_title('Latency Distribution by Variant')
        axes[0, 0].set_ylabel('Latency (s)')
        
        # Satisfaction scores
        satisfaction_data = self.df[self.df['satisfaction_score'].notna()]
        satisfaction_data.boxplot(column='satisfaction_score', by='variant_id', ax=axes[0, 1])
        axes[0, 1].set_title('Satisfaction Score by Variant')
        axes[0, 1].set_ylabel('Satisfaction (0-5)')
        
        # Token usage
        self.df.boxplot(column='tokens_output', by='variant_id', ax=axes[1, 0])
        axes[1, 0].set_title('Output Tokens by Variant')
        axes[1, 0].set_ylabel('Tokens')
        
        # Cost
        self.df.boxplot(column='cost', by='variant_id', ax=axes[1, 1])
        axes[1, 1].set_title('Cost per Query by Variant')
        axes[1, 1].set_ylabel('Cost ($)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to {save_path}")
        
        return fig

def main():
    """Run complete A/B test analysis"""
    print("="*80)
    print("A/B TEST STATISTICAL ANALYSIS")
    print("="*80)
    
    analyzer = ABTestAnalyzer()
    
    # Summary statistics
    analyzer.summary_stats()
    
    # ANOVA tests
    analyzer.compare_all_variants('latency')
    
    # Pairwise comparisons
    analyzer.pairwise_comparison('latency')
    analyzer.pairwise_comparison('satisfaction_score')
    
    # Recommendation
    results = analyzer.recommend_winner()
    
    # Visualizations
    try:
        analyzer.plot_distributions()
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
    
    print("\n✅ Analysis complete!")
    print("Check monitoring/ab_test_plots.png for visualizations")

if __name__ == "__main__":
    main()