"""
A safeguard that prevents heavy ML dependencies (torch, detoxify, spacy, presidio)
from loading on unsupported systems like Windows + Python 3.14.
"""

import sys
from unittest.mock import MagicMock


def apply_safe_imports():
    MOCK_LIBRARIES = [
        "torch",
        "detoxify",
        "spacy",
        "presidio_analyzer",
        "presidio_anonymizer",
        "thinc",
    ]

    for lib in MOCK_LIBRARIES:
        if lib not in sys.modules:
            sys.modules[lib] = MagicMock()
