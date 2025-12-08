"""
A/B Testing System for RAG Prompts
Tests multiple prompt variants and tracks performance
"""

import random
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics for A/B testing
ab_test_queries = Counter(
    "ab_test_queries_total", "A/B test queries", ["variant", "status"]
)
ab_test_latency = Histogram(
    "ab_test_latency_seconds", "Latency by variant", ["variant"]
)
ab_test_tokens = Counter(
    "ab_test_tokens_total", "Tokens by variant", ["variant", "type"]
)
ab_test_satisfaction = Histogram(
    "ab_test_satisfaction_score", "User satisfaction", ["variant"]
)
ab_test_active_variant = Gauge(
    "ab_test_active_variant", "Current active variant", ["variant"]
)


@dataclass
class PromptVariant:
    """Represents a prompt variant for testing"""

    id: str
    name: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 2048
    description: str = ""

    def to_dict(self):
        return asdict(self)


class PromptVariants:
    """Predefined prompt variants for A/B testing"""

    CONTROL = PromptVariant(
        id="control",
        name="Control (Original)",
        system_prompt="""You are an expert energy and sustainability assistant. 
Use the following pieces of context to answer the question accurately and comprehensively.

Guidelines:
- Base your answer on the provided context
- If the context doesn't contain enough information, say so clearly
- Provide detailed, well-structured answers
- Cite specific information from the context when relevant
- If asked about numerical data, be precise

Context:
{context}""",
        temperature=0.7,
        description="Original prompt with balanced approach",
    )

    CONCISE = PromptVariant(
        id="concise",
        name="Concise Variant",
        system_prompt="""You are an energy efficiency expert. Provide brief, actionable answers.

Instructions:
- Keep answers under 100 words
- Focus on key facts only
- Use bullet points when possible
- Prioritize actionable advice
- Be direct and specific

Context:
{context}""",
        temperature=0.5,
        max_tokens=512,
        description="Short, actionable responses optimized for quick reading",
    )

    DETAILED = PromptVariant(
        id="detailed",
        name="Detailed Variant",
        system_prompt="""You are a comprehensive energy and sustainability consultant. 
Provide thorough, educational answers with examples and explanations.

Guidelines:
- Explain concepts in detail with examples
- Include relevant statistics and data
- Break down complex topics step-by-step
- Provide context and background information
- Use analogies to clarify difficult concepts
- Always cite sources from the context

Context:
{context}""",
        temperature=0.8,
        max_tokens=3072,
        description="Comprehensive answers with examples and detailed explanations",
    )

    CONVERSATIONAL = PromptVariant(
        id="conversational",
        name="Conversational Variant",
        system_prompt="""You are a friendly energy advisor having a natural conversation.
Respond in a warm, approachable tone while remaining accurate.

Style:
- Use conversational language
- Include friendly phrases like "Great question!"
- Explain things simply without jargon
- Use "you" and "your" to personalize
- Add encouraging comments
- Keep it friendly but professional

Context:
{context}""",
        temperature=0.9,
        description="Friendly, conversational tone for better engagement",
    )

    TECHNICAL = PromptVariant(
        id="technical",
        name="Technical Variant",
        system_prompt="""You are a technical energy systems specialist. 
Provide precise, technical answers with industry terminology.

Approach:
- Use technical terminology accurately
- Include specific metrics and units
- Reference standards and regulations
- Provide quantitative data when available
- Focus on engineering and scientific aspects
- Assume audience has technical background

Context:
{context}""",
        temperature=0.3,
        description="Technical, precise answers for expert users",
    )

    @classmethod
    def get_all_variants(cls) -> List[PromptVariant]:
        """Get all available variants"""
        return [
            cls.CONTROL,
            cls.CONCISE,
            cls.DETAILED,
            cls.CONVERSATIONAL,
            cls.TECHNICAL,
        ]

    @classmethod
    def get_variant(cls, variant_id: str) -> Optional[PromptVariant]:
        """Get variant by ID"""
        variants = {v.id: v for v in cls.get_all_variants()}
        return variants.get(variant_id)


@dataclass
class ABTestResult:
    """Result from an A/B test query"""

    variant_id: str
    variant_name: str
    query: str
    answer: str
    latency: float
    tokens_input: int
    tokens_output: int
    cost: float
    timestamp: str
    satisfaction_score: Optional[float] = None
    retrieval_scores: Optional[List[float]] = None

    def to_dict(self):
        return asdict(self)


class ABTestingEngine:
    """
    A/B Testing engine for prompt variants.

    Features:
    - Random variant assignment
    - Performance tracking per variant
    - Statistical comparison
    - Result logging
    """

    def __init__(
        self,
        variants: Optional[List[PromptVariant]] = None,
        traffic_split: Optional[Dict[str, float]] = None,
        results_file: str = "monitoring/ab_test_results.jsonl",
    ):
        """
        Initialize A/B testing engine.

        Args:
            variants: List of prompt variants to test (defaults to all)
            traffic_split: Dict of variant_id -> percentage (e.g., {"control": 0.5, "concise": 0.5})
            results_file: Path to store test results
        """
        self.variants = variants or PromptVariants.get_all_variants()
        self.variant_map = {v.id: v for v in self.variants}

        # Default to equal split
        if traffic_split is None:
            split = 1.0 / len(self.variants)
            self.traffic_split = {v.id: split for v in self.variants}
        else:
            self.traffic_split = traffic_split

        self.results_file = results_file
        self._ensure_results_file()

        # Update Prometheus gauge
        for variant in self.variants:
            ab_test_active_variant.labels(variant=variant.id).set(1)

        logger.info(f"A/B Testing initialized with {len(self.variants)} variants")
        logger.info(f"Traffic split: {self.traffic_split}")

    def _ensure_results_file(self):
        """Create results file if not exists"""
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        if not os.path.exists(self.results_file):
            open(self.results_file, "w").close()

    def assign_variant(self, user_id: Optional[str] = None) -> PromptVariant:
        """
        Assign a variant to a user/query.

        Args:
            user_id: Optional user ID for consistent assignment

        Returns:
            Assigned prompt variant
        """
        if user_id:
            # Consistent hashing for same user
            hash_val = hash(user_id) % 100 / 100.0
        else:
            # Random assignment
            hash_val = random.random()

        # Select variant based on traffic split
        cumulative = 0
        for variant_id, percentage in self.traffic_split.items():
            cumulative += percentage
            if hash_val <= cumulative:
                variant = self.variant_map[variant_id]
                logger.info(f"Assigned variant: {variant.name} ({variant.id})")
                return variant

        # Fallback to control
        return self.variant_map.get("control", self.variants[0])

    def log_result(self, result: ABTestResult):
        """Log A/B test result"""
        # Update Prometheus metrics
        ab_test_queries.labels(variant=result.variant_id, status="success").inc()

        ab_test_latency.labels(variant=result.variant_id).observe(result.latency)

        ab_test_tokens.labels(variant=result.variant_id, type="input").inc(
            result.tokens_input
        )

        ab_test_tokens.labels(variant=result.variant_id, type="output").inc(
            result.tokens_output
        )

        if result.satisfaction_score:
            ab_test_satisfaction.labels(variant=result.variant_id).observe(
                result.satisfaction_score
            )

        # Write to file
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

        logger.info(f"Logged result for variant {result.variant_id}")

    def get_variant_stats(self, variant_id: str) -> Dict:
        """Get statistics for a specific variant"""
        results = self.load_results()
        variant_results = [r for r in results if r["variant_id"] == variant_id]

        if not variant_results:
            return {
                "variant_id": variant_id,
                "total_queries": 0,
                "avg_latency": 0,
                "avg_tokens": 0,
                "total_cost": 0,
                "avg_satisfaction": 0,
            }

        return {
            "variant_id": variant_id,
            "total_queries": len(variant_results),
            "avg_latency": sum(r["latency"] for r in variant_results)
            / len(variant_results),
            "avg_tokens": sum(
                r["tokens_input"] + r["tokens_output"] for r in variant_results
            )
            / len(variant_results),
            "total_cost": sum(r["cost"] for r in variant_results),
            "avg_satisfaction": (
                sum(
                    r.get("satisfaction_score", 0)
                    for r in variant_results
                    if r.get("satisfaction_score")
                )
                / len([r for r in variant_results if r.get("satisfaction_score")])
                if any(r.get("satisfaction_score") for r in variant_results)
                else 0
            ),
        }

    def load_results(self) -> List[Dict]:
        """Load all test results"""
        results = []
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        return results

    def get_comparison_report(self) -> Dict:
        """Generate comparison report for all variants"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "variants": {},
            "winner": None,
        }

        best_score = 0
        best_variant = None

        for variant in self.variants:
            stats = self.get_variant_stats(variant.id)
            report["variants"][variant.id] = {
                "name": variant.name,
                "description": variant.description,
                "stats": stats,
                "temperature": variant.temperature,
                "max_tokens": variant.max_tokens,
            }

            # Simple scoring: balance speed, cost, and satisfaction
            if stats["total_queries"] > 0:
                score = (
                    (1 / (stats["avg_latency"] + 0.1)) * 0.3  # Speed (30%)
                    + (1 / (stats["total_cost"] / stats["total_queries"] + 0.001))
                    * 0.3  # Cost efficiency (30%)
                    + stats["avg_satisfaction"] * 0.4  # Satisfaction (40%)
                )

                if score > best_score:
                    best_score = score
                    best_variant = variant.id

        report["winner"] = best_variant
        return report


def create_ab_testing_engine(
    enabled_variants: Optional[List[str]] = None,
    traffic_split: Optional[Dict[str, float]] = None,
) -> ABTestingEngine:
    """
    Factory function to create A/B testing engine.

    Args:
        enabled_variants: List of variant IDs to enable (None = all)
        traffic_split: Custom traffic split

    Returns:
        Configured ABTestingEngine
    """
    if enabled_variants:
        variants = [
            PromptVariants.get_variant(vid)
            for vid in enabled_variants
            if PromptVariants.get_variant(vid)
        ]
    else:
        variants = PromptVariants.get_all_variants()

    return ABTestingEngine(variants=variants, traffic_split=traffic_split)
