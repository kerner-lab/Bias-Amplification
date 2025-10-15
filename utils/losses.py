import torch

bce_loss = torch.nn.BCELoss()


def ModifiedBCELoss(y_pred, y_true):
    return 1 / bce_loss(y_pred, y_true)
