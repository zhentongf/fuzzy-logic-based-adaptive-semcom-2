import torch
import torch.nn.functional as F

def compute_psnr(x, x_hat):
    mse = F.mse_loss(x_hat, x)
    if mse == 0:
        return 100
    return 10 * torch.log10(1.0 / mse)

def compute_accuracy(classifier, x, x_hat):
    with torch.no_grad():
        y_pred = classifier(x).argmax(dim=1)
        y_hat = classifier(x_hat).argmax(dim=1)
        acc = (y_pred == y_hat).float().mean()
    return acc.item()