import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import mlflow
from typing import List, Dict
from tqdm import tqdm
from fastembed import TextEmbedding
from dotenv import load_dotenv
from src.rag.config import (
    INDEX_PATH,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    FASTEMBED_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TOP_K,
)
from src.rag.custom_retriever import create_retriever
from src.rag.rag_chain import EnhancedRAGChain
from experiments.prompts import PROMPTS


# --- 0. ENV CONFIG ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# --- 1. SETUP PATHS & IMPORTS ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

load_dotenv()

# Import Config (Triggers LangSmith setup if keys are present)


# Import RAG Components from the NEW architecture


# --- 2. CONFIGURATION ---
MLFLOW_TRACKING_URI = "http://13.220.125.38:8000/"
EXPERIMENT_NAME = "RAG_Prompt_Engineering"
EVAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "eval.jsonl")
REPORT_PATH = os.path.join(PROJECT_ROOT, "prompt_report.md")

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Silent the noisy logs from src to keep experiment output clean
logging.getLogger("src.rag.custom_retriever").setLevel(logging.WARNING)
logging.getLogger("src.rag.rag_chain").setLevel(logging.WARNING)


class ExperimentEvaluator:
    def __init__(self):
        """Initialize evaluation tools."""
        logger.info("Initializing Evaluator...")

        # 1. Standalone embedding model for similarity metric
        self.eval_embedder = TextEmbedding(model_name=FASTEMBED_MODEL)

        # 2. Standalone LLM for 'Judge' metric
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.judge_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.0
        )

        # 3. Shared Retriever
        # We reuse the factory from src/rag/custom_retriever.py to match production logic
        self.shared_retriever = create_retriever(
            index_path=INDEX_PATH, embedding_model=FASTEMBED_MODEL, k=TOP_K
        )
        logger.info("✓ Evaluator Initialized")

    def run_strategy(
        self, strategy_name: str, system_prompt: str, eval_data: List[Dict]
    ):
        """Run the experiment for a specific prompt strategy."""

        # Initialize Chain with the specific System Prompt
        # This matches src/rag/rag_chain.py logic
        chain = EnhancedRAGChain(
            retriever=self.shared_retriever,
            llm_model=GEMINI_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            google_api_key=GOOGLE_API_KEY,
            system_prompt=system_prompt,
        )

        results = []
        scores = []
        similarities = []

        for item in tqdm(eval_data, desc=f"Testing {strategy_name}"):
            query = item["question"]
            ground_truth = item["ground_truth"]

            try:
                # 1. Invoke Chain
                # EnhancedRAGChain.invoke returns a dict with 'answer' key
                response = chain.invoke(query)
                generated_answer = response["answer"]

                # 2. Metrics
                cos_sim = self._calculate_cosine_similarity(
                    generated_answer, ground_truth
                )
                quality_score = self._llm_judge(query, ground_truth, generated_answer)

                # 3. Record
                results.append(
                    {
                        "strategy": strategy_name,
                        "question": query,
                        "ground_truth": ground_truth,
                        "generated_answer": generated_answer,
                        "cosine_similarity": cos_sim,
                        "quality_score": quality_score,
                    }
                )
                scores.append(quality_score)
                similarities.append(cos_sim)

            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")

        return results, np.mean(scores), np.mean(similarities)

    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        vec1 = list(self.eval_embedder.embed([text1]))[0]
        vec2 = list(self.eval_embedder.embed([text2]))[0]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _llm_judge(self, question: str, ground_truth: str, generated: str) -> int:
        """Ask LLM to score the answer 1-5."""
        judge_prompt = f"""
        You are an impartial judge evaluating the quality of an AI-generated answer.
        
        Question: {question}
        Ground Truth: {ground_truth}
        Generated Answer: {generated}
        
        Rate the Generated Answer on a scale of 1 to 5 based on Factuality and Completeness.
        Return ONLY the integer number.
        """
        try:
            res = self.judge_llm.invoke(judge_prompt)
            score = int("".join(filter(str.isdigit, res.content)))
            return max(1, min(5, score))
        except Exception:
            return 3


def load_eval_data():
    data = []
    if not os.path.exists(EVAL_DATA_PATH):
        raise FileNotFoundError(f"Evaluation data not found at {EVAL_DATA_PATH}")
    with open(EVAL_DATA_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    evaluator = ExperimentEvaluator()
    eval_data = load_eval_data()
    all_results = []

    print(
        f"🚀 Starting Prompt Engineering Experiment on {len(eval_data)} test cases..."
    )

    for strategy_name, prompt_template in PROMPTS.items():
        print(f"\n🧪 Strategy: {strategy_name}")

        with mlflow.start_run(run_name=f"Strategy_{strategy_name}"):
            # Log params
            mlflow.log_param("strategy", strategy_name)
            mlflow.log_param("system_prompt", prompt_template)

            # Run
            results, avg_score, avg_sim = evaluator.run_strategy(
                strategy_name, prompt_template, eval_data
            )
            all_results.extend(results)

            # Log metrics
            mlflow.log_metric("avg_quality_score", avg_score)
            mlflow.log_metric("avg_cosine_similarity", avg_sim)

            print(f"   -> Score: {avg_score:.2f}/5 | Similarity: {avg_sim:.4f}")

    # Generate Report
    generate_report(all_results)


def generate_report(results):
    df = pd.DataFrame(results)

    report_content = f"""# 🧪 Prompt Engineering Report

## 1. Summary
- **Date**: {time.strftime("%Y-%m-%d")}
- **Model**: {GEMINI_MODEL}
- **Strategies Tested**: {', '.join(df['strategy'].unique())}

## 2. Quantitative Results
| Strategy | Avg Cosine Similarity | Avg Quality Score (1-5) |
|----------|-----------------------|-------------------------|
"""
    summary = df.groupby("strategy")[["cosine_similarity", "quality_score"]].mean()
    for strategy, row in summary.iterrows():
        report_content += f"| {strategy} | {row['cosine_similarity']:.4f} | {row['quality_score']:.2f} |\n"

    report_content += "\n## 3. Qualitative Analysis & Examples\n"

    for strategy in df["strategy"].unique():
        report_content += f"\n### Strategy: {strategy}\n"
        subset = df[df["strategy"] == strategy].head(2)
        for _, row in subset.iterrows():
            report_content += f"**Q:** {row['question']}\n"
            report_content += f"**Generated:** {row['generated_answer']}\n"
            report_content += f"**Score:** {row['quality_score']}/5\n"
            report_content += "---\n"

    # Fix: Use UTF-8 encoding to support emojis and special characters
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n📄 Report generated at: {REPORT_PATH}")


if __name__ == "__main__":
    main()
