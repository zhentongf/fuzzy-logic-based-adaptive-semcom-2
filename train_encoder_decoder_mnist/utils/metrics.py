import torch
import torch.nn.functional as F


def compute_psnr(x, x_hat):

    mse = F.mse_loss(x_hat, x)

    if mse.item() == 0:
        return torch.tensor(100.0)

    return 10 * torch.log10(1.0 / mse)


def compute_accuracy(classifier, x, x_hat):

    with torch.no_grad():

        y_true = classifier(x).argmax(dim=1)

        y_pred = classifier(x_hat).argmax(dim=1)

        acc = (y_true == y_pred).float().mean()

    return acc.item()