import logging
import re
from typing import Dict, Any
from detoxify import Detoxify

logger = logging.getLogger(__name__)


class OutputModerator:
    """Moderates model outputs for safety and quality."""

    # Hallucination indicators
    HALLUCINATION_PHRASES = [
        r"i\s+(think|believe|assume|guess)",
        r"(probably|maybe|perhaps|possibly)",
        r"i\s+don\'?t\s+(know|remember)",
        r"as\s+an\s+ai\s+(language\s+)?model",
        r"i\s+(cannot|can\'?t)\s+(verify|confirm)",
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Output Moderator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enable_toxicity = config.get("enable_toxicity_filter", True)
        self.enable_hallucination = config.get("enable_hallucination_filter", True)
        self.toxicity_threshold = config.get("toxicity_threshold", 0.7)
        self.max_length = config.get("max_output_length", 2048)

        # Initialize toxicity detector
        if self.enable_toxicity:
            try:
                self.toxicity_model = Detoxify("original")
                logger.info("Toxicity detection initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize toxicity detection: {e}")
                self.enable_toxicity = False

    def moderate(self, model_output: str) -> Dict[str, Any]:
        """
        Moderate model output against all checks.

        Args:
            model_output: The model's generated output

        Returns:
            Dictionary with moderation results
        """
        result = {
            "passed": True,
            "violations": [],
            "sanitized_output": model_output,
            "toxicity_scores": {},
            "hallucination_detected": False,
        }

        # Check output length
        if len(model_output) > self.max_length:
            result["passed"] = False
            result["violations"].append(
                {
                    "type": "OUTPUT_LENGTH_EXCEEDED",
                    "message": f"Output length {len(model_output)} exceeds maximum {self.max_length}",
                }
            )
            result["sanitized_output"] = model_output[: self.max_length]

        # Check for empty output
        if not model_output.strip():
            result["passed"] = False
            result["violations"].append(
                {"type": "EMPTY_OUTPUT", "message": "Output cannot be empty"}
            )
            return result

        # Toxicity detection
        if self.enable_toxicity:
            toxicity_result = self._detect_toxicity(model_output)
            result["toxicity_scores"] = toxicity_result["scores"]

            if toxicity_result["is_toxic"]:
                result["passed"] = False
                result["violations"].append(
                    {
                        "type": "TOXICITY_DETECTED",
                        "message": "Output contains toxic content",
                        "scores": toxicity_result["scores"],
                    }
                )

        # Hallucination detection
        if self.enable_hallucination:
            hallucination_result = self._detect_hallucination(model_output)
            result["hallucination_detected"] = hallucination_result["detected"]

            if hallucination_result["detected"]:
                result["violations"].append(
                    {
                        "type": "POTENTIAL_HALLUCINATION",
                        "message": "Output may contain hallucinated content",
                        "indicators": hallucination_result["indicators"],
                        "severity": "WARNING",  # Don't block, just warn
                    }
                )

        return result

    def _detect_toxicity(self, text: str) -> Dict[str, Any]:
        """
        Detect toxicity in text.

        Args:
            text: Output text

        Returns:
            Dictionary with toxicity scores
        """
        try:
            scores = self.toxicity_model.predict(text)

            # Convert numpy types to Python types
            scores = {k: float(v) for k, v in scores.items()}

            # Check if any score exceeds threshold
            is_toxic = any(score > self.toxicity_threshold for score in scores.values())

            return {"is_toxic": is_toxic, "scores": scores}

        except Exception as e:
            logger.error(f"Error in toxicity detection: {e}", exc_info=True)
            return {"is_toxic": False, "scores": {}}

    def _detect_hallucination(self, text: str) -> Dict[str, Any]:
        """
        Detect potential hallucination indicators in text.

        Args:
            text: Output text

        Returns:
            Dictionary with hallucination detection results
        """
        detected_indicators = []
        text_lower = text.lower()

        for pattern in self.HALLUCINATION_PHRASES:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                detected_indicators.append({"pattern": pattern, "matches": matches})

        return {
            "detected": len(detected_indicators) > 0,
            "indicators": detected_indicators,
        }
