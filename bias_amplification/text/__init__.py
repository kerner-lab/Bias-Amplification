"""
Text processing module for bias amplification.

This module requires optional dependencies. Install with:
    pip install bias-amplification[text]

Required packages:
- torchtext: For text tokenization
- transformers: For BERT embeddings
- gensim: For GloVe embeddings
- sentence-transformers: For sentence embeddings
- nltk: For NLTK tokenizer
- tqdm: For progress bars
"""

# Check dependencies BEFORE attempting any imports
_REQUIRED_PACKAGES = {
    "torchtext": "torchtext",
    "transformers": "transformers",
    "gensim": "gensim",
    "sentence_transformers": "sentence_transformers",
    "nltk": "nltk",
    "tqdm": "tqdm"
}

_missing_packages = []
for display_name, import_name in _REQUIRED_PACKAGES.items():
    try:
        __import__(import_name)
    except ImportError:
        _missing_packages.append(display_name)

if _missing_packages:
    raise ImportError(
        f"Text module dependencies not installed: {', '.join(_missing_packages)}\n"
        f"Install with: pip install 'bias-amplification[text]'\n\n"
        f"Attempting to import from bias_amplification.text requires the [text] extra dependencies."
    )


from .metrics import LIC, DBAC
from .utils.text import CaptionProcessor
from .attacker_models import LSTM_ANN_Model, RNN_ANN_Model, SimpleTransformer

__all__ = [
    "LIC",
    "DBAC",
    "CaptionProcessor",
    "LSTM_ANN_Model",
    "RNN_ANN_Model",
    "SimpleTransformer",
]