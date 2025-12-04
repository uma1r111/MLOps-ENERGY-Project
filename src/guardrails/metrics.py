"""
Prometheus metrics for guardrail monitoring.
"""

from prometheus_client import Counter, Histogram, Gauge
import logging

logger = logging.getLogger(__name__)

# Metrics definitions
guardrail_validations_total = Counter(
    'guardrail_validations_total',
    'Total number of guardrail validations',
    ['validation_type', 'result']
)

guardrail_violations_total = Counter(
    'guardrail_violations_total',
    'Total number of guardrail violations',
    ['violation_type']
)

guardrail_latency_seconds = Histogram(
    'guardrail_latency_seconds',
    'Guardrail processing latency in seconds',
    ['operation']
)

pii_detections_total = Counter(
    'pii_detections_total',
    'Total number of PII detections',
    ['entity_type']
)

toxicity_score = Histogram(
    'toxicity_score',
    'Toxicity scores from output moderation',
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)


class GuardrailMetrics:
    """Helper class for recording guardrail metrics."""
    
    @staticmethod
    def record_validation(validation_type: str, passed: bool):
        """Record a validation event."""
        result = 'passed' if passed else 'failed'
        guardrail_validations_total.labels(
            validation_type=validation_type,
            result=result
        ).inc()
    
    @staticmethod
    def record_violation(violation_type: str):
        """Record a violation."""
        guardrail_violations_total.labels(
            violation_type=violation_type
        ).inc()
    
    @staticmethod
    def record_latency(operation: str, latency_ms: float):
        """Record operation latency."""
        guardrail_latency_seconds.labels(
            operation=operation
        ).observe(latency_ms / 1000.0)
    
    @staticmethod
    def record_pii_detection(entity_type: str):
        """Record PII detection."""
        pii_detections_total.labels(
            entity_type=entity_type
        ).inc()
    
    @staticmethod
    def record_toxicity_score(score: float):
        """Record toxicity score."""
        toxicity_score.observe(score)