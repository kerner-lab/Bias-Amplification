"""
Utility functions for bias amplification analysis.
"""

from .datacreator import dataCreator, validate_error_percent
from .config import (
    DEFAULT_TEST_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_METRIC,
    DEFAULT_NUM_TRIALS,
    DEFAULT_PREDICTION_THRESHOLD,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OPTIMIZER,
    DEFAULT_SCHEDULER,
    DEFAULT_AGGREGATION_METHOD,
    EPOCH_LOG_INTERVAL,
    normalise,
)
from .losses import ModifiedBCELoss, ModifiedMSELoss

__all__ = [
    "dataCreator",
    "validate_error_percent",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EVAL_METRIC",
    "DEFAULT_NUM_TRIALS",
    "DEFAULT_PREDICTION_THRESHOLD",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_OPTIMIZER",
    "DEFAULT_SCHEDULER",
    "DEFAULT_AGGREGATION_METHOD",
    "EPOCH_LOG_INTERVAL",
    "normalise",
    "ModifiedBCELoss",
    "ModifiedMSELoss",
]
