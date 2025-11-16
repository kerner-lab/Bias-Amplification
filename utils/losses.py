import torch

bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss()


def ModifiedBCELoss(y_pred, y_true):
    return 1 / bce_loss(y_pred, y_true)

def ModifiedMSELoss(y_pred, y_true):
    return 1 / mse_loss(y_pred, y_true)
