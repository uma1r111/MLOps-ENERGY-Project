import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_guardrails_imports():
    """Test that guardrails module can be imported."""
    try:
        import src.guardrails

        assert src.guardrails is not None
    except ImportError as e:
        assert False, f"Failed to import guardrails: {e}"
