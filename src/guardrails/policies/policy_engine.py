import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PolicyEngine:
    """Enforces custom policies for inputs and outputs."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Policy Engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.block_on_violation = config.get("block_on_violation", True)
        self.log_all_events = config.get("log_all_events", True)

    def check_input_policy(
        self, user_input: str, validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check input against custom policies.

        Args:
            user_input: The user's input text
            validation_result: Results from input validation

        Returns:
            Dictionary with policy check results
        """
        result = {"passed": True, "violations": []}

        # If PII was detected, log it
        if validation_result.get("pii_detected"):
            logger.warning(
                f"PII detected in input: {validation_result['pii_detected']}"
            )

        # If prompt injection detected and blocking is enabled
        if validation_result.get("injection_detected") and self.block_on_violation:
            logger.error("Prompt injection attempt blocked")

        return result

    def check_output_policy(
        self, model_output: str, moderation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check output against custom policies.

        Args:
            model_output: The model's output text
            moderation_result: Results from output moderation

        Returns:
            Dictionary with policy check results
        """
        result = {"passed": True, "violations": []}

        # Custom policy: Block outputs with high toxicity if blocking enabled
        toxicity_scores = moderation_result.get("toxicity_scores", {})
        if toxicity_scores and self.block_on_violation:
            max_toxicity = max(toxicity_scores.values()) if toxicity_scores else 0
            if max_toxicity > 0.8:  # Stricter threshold for blocking
                result["passed"] = False
                result["violations"].append(
                    {
                        "type": "HIGH_TOXICITY_BLOCKED",
                        "message": "Output blocked due to high toxicity",
                        "max_score": max_toxicity,
                    }
                )

        return result
