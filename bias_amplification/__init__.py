"""
Directional Predictability Bias Amplification Library

A library used as metric for measuring bias amplification and leakage information
in machine learning models.

Main modules:
- metrics: Predictability metrics (Leakage, DPA, LIC)
- utils: Utility functions for data creation and configuration
- attacker_models: Neural network models for attacker simulation
- text: Text processing modules (optional, install with: pip install bias-amplification[text])
"""

__version__ = "0.1.0"

from .metrics.PredMetrics import BasePredictabilityMetric, Leakage, DPA
from .metrics.CoOccurMetrics import BaseCoOccurMetric, BA_MALS, DBA, MDBA

__all__ = [
    "Leakage",
    "DPA",
    "BasePredictabilityMetric",
    "BaseCoOccurMetric",
    "BA_MALS",
    "DBA",
    "MDBA",
    "__version__",
]

# Optional text module imports - lazy import with dependency check
def __getattr__(name):
    """Lazy import for optional text modules."""
    if name in ("LIC", "DBAC", "CaptionProcessor", "LSTM_ANN_Model", "RNN_ANN_Model", "SimpleTransformer"):
        try:
            if name == "LIC":
                from .text.metrics import LIC
                return LIC
            elif name == "DBAC":
                from .text.metrics import DBAC
                return DBAC
            elif name == "CaptionProcessor":
                from .text.utils.text import CaptionProcessor
                return CaptionProcessor
            elif name == "LSTM_ANN_Model":
                from .text.attacker_models import LSTM_ANN_Model
                return LSTM_ANN_Model
            elif name == "RNN_ANN_Model":
                from .text.attacker_models import RNN_ANN_Model
                return RNN_ANN_Model
            elif name == "SimpleTransformer":
                from .text.attacker_models import SimpleTransformer
                return SimpleTransformer
        except ImportError as e:
            # Re-raise with helpful message
            raise ImportError(
                f"Text module '{name}' requires additional dependencies.\n"
                f"Install with: pip install bias-amplification[text]\n"
                f"Original error: {e}"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")