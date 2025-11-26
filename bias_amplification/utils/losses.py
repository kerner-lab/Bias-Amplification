import torch

bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss()


def ModifiedBCELoss(y_pred, y_true):
    """
    This function computes the modified version of the BCE loss.
    The modified version of cross-entropy loss gives high values for better performance.

    Parameters
    ----------
    y_pred: torch.Tensor
        The predicted values.
    y_true: torch.Tensor
        The true values.

    Returns
    -------
    loss: torch.Tensor
        The modified BCE loss.
    """
    return 1 / bce_loss(y_pred, y_true)

def ModifiedMSELoss(y_pred, y_true):
    """
    This function computes the modified version of the MSE loss.
    The modified version of MSE loss gives high values for better performance.

    Parameters
    ----------
    y_pred: torch.Tensor
        The predicted values.
    y_true: torch.Tensor
        The true values.

    Returns
    -------
    loss: torch.Tensor
        The modified MSE loss.
    """
    return 1 / mse_loss(y_pred, y_true)
