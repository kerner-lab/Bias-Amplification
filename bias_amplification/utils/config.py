import torch
from typing import Union


# ============================================================================
# CONSTANTS
# ============================================================================
DEFAULT_TEST_SIZE = 0.2
EPOCH_LOG_INTERVAL = 10
DEFAULT_PREDICTION_THRESHOLD = 0.5
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
DEFAULT_EVAL_METRIC = "mse"
DEFAULT_OPTIMIZER = "adam"
DEFAULT_SCHEDULER = "cosine"
DEFAULT_AGGREGATION_METHOD = "mean"
DEFAULT_NUM_TRIALS = 10
# ============================================================================
# UTILS FUNCTIONS
# ============================================================================
def normalise(value: Union[float, int, torch.Tensor]) -> Union[float, int, torch.Tensor]:
    """
    This function normalises a value to be between 0 and 1.

    Parameters
    ----------
    value: float, int, or torch.Tensor
        The value to normalise.

    Returns
    -------
    value: float, int, or torch.Tensor
        The normalised value.
    """
    if value > 1:
        value = value / 100
    return value

