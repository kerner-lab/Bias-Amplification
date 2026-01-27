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

try:
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
except ImportError as e:
    import sys
    raise ImportError(
        f"Text module dependencies not installed. "
        f"Install with: pip install 'bias-amplification[text]'\n"
        f"Original error: {e}"
    ) from e