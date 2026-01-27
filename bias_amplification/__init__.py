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

# Optional text module imports
try:
    from .text.metrics import LIC, DBAC
    __all__.extend(["LIC", "DBAC"])
except ImportError as e:
    # Text module not installed
    _text_import_error = e
    def _raise_text_import_error():
        raise ImportError(
            "Text module requires additional dependencies. "
            "Install with: pip install bias-amplification[text]"
        ) from _text_import_error
    
    # Create placeholder classes that raise helpful errors
    class LIC:
        def __init__(self, *args, **kwargs):
            _raise_text_import_error()
    
    class DBAC:
        def __init__(self, *args, **kwargs):
            _raise_text_import_error()