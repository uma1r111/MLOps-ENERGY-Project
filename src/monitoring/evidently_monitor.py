"""
Evidently Monitoring for RAG Data Drift
Monitors embedding distribution and retrieval quality
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDistributionMetric
)

sys.path.append(str(Path(__file__).parent.parent))
from src.rag.config import INDEX_PATH, FASTEMBED_MODEL
from fastembed import TextEmbedding

class EmbeddingDriftMonitor:
    """Monitor drift in document embeddings"""
    
    def __init__(self, index_path: str = INDEX_PATH):
        self.index_path = index_path
        self.embedding_model = TextEmbedding(model_name=FASTEMBED_MODEL)
        self.reference_data = None
        
    def load_reference_embeddings(self) -> pd.DataFrame:
        """Load original embeddings as reference"""
        print("Loading reference embeddings...")
        
        # Load documents
        with open(os.path.join(self.index_path, "documents.pkl"), "rb") as f:
            docs = pickle.load(f)
        
        # Get embeddings
        texts = [doc.page_content[:500] for doc in docs[:100]]  # Sample
        embeddings = list(self.embedding_model.embed(texts))
        embeddings_array = np.array(embeddings)
        
        # Create DataFrame with embedding dimensions
        df = pd.DataFrame(
            embeddings_array,
            columns=[f"dim_{i}" for i in range(embeddings_array.shape[1])]
        )
        
        # Add metadata
        df['source'] = [doc.metadata.get('source', 'unknown') for doc in docs[:100]]
        df['chunk_length'] = [len(doc.page_content) for doc in docs[:100]]
        
        print(f"✓ Loaded {len(df)} reference embeddings")
        return df
    
    def generate_current_embeddings(self, new_texts: List[str]) -> pd.DataFrame:
        """Generate current embeddings for comparison"""
        print("Generating current embeddings...")
        
        embeddings = list(self.embedding_model.embed(new_texts))
        embeddings_array = np.array(embeddings)
        
        df = pd.DataFrame(
            embeddings_array,
            columns=[f"dim_{i}" for i in range(embeddings_array.shape[1])]
        )
        
        df['source'] = ['new_data'] * len(new_texts)
        df['chunk_length'] = [len(t) for t in new_texts]
        
        print(f"✓ Generated {len(df)} current embeddings")
        return df
    
    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Report:
        """Detect drift between reference and current data"""
        print("Detecting drift...")
        
        # Create Evidently report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDistributionMetric(column_name='chunk_length')
        ])
        
        # Run report
        report.run(
            reference_data=reference_df,
            current_data=current_df
        )
        
        print("✓ Drift detection complete")
        return report
    
    def generate_report(self, output_path: str = "monitoring/evidently_report.html"):
        """Generate full drift monitoring report"""
        print("\n" + "="*60)
        print("Evidently Drift Monitoring Report")
        print("="*60)
        
        # Load reference data
        reference_df = self.load_reference_embeddings()
        
        # Simulate current data (in production, this would be new queries)
        # For demo: add noise to reference data
        current_df = reference_df.copy()
        
        # Add some drift by perturbing embeddings
        embedding_cols = [col for col in current_df.columns if col.startswith('dim_')]
        current_df[embedding_cols] += np.random.normal(0, 0.01, current_df[embedding_cols].shape)
        
        # Generate report
        report = self.detect_drift(reference_df, current_df)
        
        # Save HTML report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)
        
        print(f"\n✓ Report saved to: {output_path}")
        print(f"  Open in browser to view drift analysis")
        print("="*60)
        
        return report


class RetrievalQualityMonitor:
    """Monitor retrieval quality metrics over time"""
    
    def __init__(self):
        self.metrics_file = "monitoring/retrieval_metrics.csv"
        self.ensure_metrics_file()
    
    def ensure_metrics_file(self):
        """Create metrics file if not exists"""
        os.makedirs("monitoring", exist_ok=True)
        
        if not os.path.exists(self.metrics_file):
            df = pd.DataFrame(columns=[
                'timestamp', 'query', 'avg_score', 'min_score', 
                'max_score', 'num_results', 'latency'
            ])
            df.to_csv(self.metrics_file, index=False)
    
    def log_retrieval(
        self, 
        query: str, 
        scores: List[float], 
        latency: float
    ):
        """Log retrieval metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'query': query[:100],
            'avg_score': np.mean(scores) if scores else 0,
            'min_score': np.min(scores) if scores else 0,
            'max_score': np.max(scores) if scores else 0,
            'num_results': len(scores),
            'latency': latency
        }
        
        df = pd.DataFrame([metrics])
        df.to_csv(self.metrics_file, mode='a', header=False, index=False)
    
    def generate_quality_report(self) -> Report:
        """Generate quality drift report"""
        print("Generating retrieval quality report...")
        
        df = pd.read_csv(self.metrics_file)
        
        if len(df) < 2:
            print("⚠️  Not enough data for quality report")
            return None
        
        # Split into reference and current
        split_idx = len(df) // 2
        reference_df = df.iloc[:split_idx].copy()
        current_df = df.iloc[split_idx:].copy()
        
        # Generate report
        report = Report(metrics=[
            DataDriftPreset(),
            ColumnDistributionMetric(column_name='avg_score'),
            ColumnDistributionMetric(column_name='latency')
        ])
        
        report.run(
            reference_data=reference_df,
            current_data=current_df
        )
        
        report.save_html("monitoring/retrieval_quality_report.html")
        print("✓ Quality report saved")
        
        return report


def main():
    """Generate all monitoring reports"""
    print("\n" + "="*60)
    print("RAG Monitoring Suite - Evidently Reports")
    print("="*60)
    
    # 1. Embedding Drift Monitor
    print("\n1. Embedding Drift Analysis")
    print("-"*60)
    drift_monitor = EmbeddingDriftMonitor()
    drift_monitor.generate_report()
    
    # 2. Retrieval Quality Monitor (demo data)
    print("\n2. Retrieval Quality Analysis")
    print("-"*60)
    quality_monitor = RetrievalQualityMonitor()
    
    # Generate demo data
    for i in range(20):
        quality_monitor.log_retrieval(
            query=f"Sample query {i}",
            scores=[0.7 + np.random.random()*0.2 for _ in range(3)],
            latency=0.5 + np.random.random()
        )
    
    quality_monitor.generate_quality_report()
    
    print("\n" + "="*60)
    print("✅ All monitoring reports generated!")
    print("="*60)
    print("\nGenerated reports:")
    print("  1. monitoring/evidently_report.html")
    print("  2. monitoring/retrieval_quality_report.html")
    print("\nOpen these files in your browser to view the analysis.")
    print("="*60)


if __name__ == "__main__":
    main()