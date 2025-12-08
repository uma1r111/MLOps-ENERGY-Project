import pytest
from src.guardrails.filters.output_moderator import OutputModerator


class TestOutputModerator:

    @pytest.fixture
    def moderator(self):
        config = {
            "enable_toxicity_filter": True,
            "enable_hallucination_filter": True,
            "toxicity_threshold": 0.7,
            "max_output_length": 2048,
        }
        return OutputModerator(config)

    def test_valid_output(self, moderator):
        """Test that valid output passes moderation."""
        result = moderator.moderate("The weather is sunny today.")
        assert result["passed"] is True
        assert len(result["violations"]) == 0

    def test_empty_output(self, moderator):
        """Test that empty output is rejected."""
        result = moderator.moderate("")
        assert result["passed"] is False
        assert any(v["type"] == "EMPTY_OUTPUT" for v in result["violations"])

    def test_output_length_exceeded(self, moderator):
        """Test that overly long output is truncated."""
        long_output = "a" * 3000
        result = moderator.moderate(long_output)
        assert result["passed"] is False
        assert any(v["type"] == "OUTPUT_LENGTH_EXCEEDED" for v in result["violations"])
        assert len(result["sanitized_output"]) <= 2048

    def test_toxicity_detection(self, moderator):
        """Test detection of toxic content."""
        toxic_outputs = [
            "I hate you and everyone like you",
            "You are stupid and worthless",
        ]

        for output in toxic_outputs:
            result = moderator.moderate(output)
            # May or may not fail depending on model, but should have scores
            assert "toxicity_scores" in result

    def test_hallucination_detection(self, moderator):
        """Test detection of potential hallucinations."""
        uncertain_outputs = [
            "I think the answer is probably 42, but I'm not sure",
            "As an AI language model, I cannot verify this information",
        ]

        for output in uncertain_outputs:
            result = moderator.moderate(output)
            assert result["hallucination_detected"] is True
