"""
Directional Predictability Bias Amplification Library

A library used as metric for measuring bias amplification and leakage information
in machine learning models.

Main modules:
- metrics: Predictability metrics (Leakage, DPA, LIC)
- utils: Utility functions for data creation and configuration
- attacker_models: Neural network models for attacker simulation
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
