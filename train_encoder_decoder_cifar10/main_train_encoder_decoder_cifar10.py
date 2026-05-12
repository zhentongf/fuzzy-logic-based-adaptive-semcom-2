import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import csv

from train_encoder_decoder_cifar10.models.encoder import Encoder
from train_encoder_decoder_cifar10.models.decoder import Decoder
from train_encoder_decoder_cifar10.models.channel import add_awgn_noise

from train_encoder_decoder_cifar10.utils.metrics import compute_psnr, compute_accuracy
from train_encoder_decoder_cifar10.utils.logger import log_csv
from train_encoder_decoder_cifar10.utils.time_utils import estimate_total_time

def get_config(main_path):
    config = {
        "dataset_path": os.path.join(main_path, "datasets/CIFAR10"),
        "batch_size": 128,
        "num_workers": 0,
        "epochs": 50,
        "lr": 1e-3
    }
    return config

# ---- CLASSIFIER ----
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---- TRAIN FUNCTION ----
def train(main_path, mode="fixed"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config(main_path)

    from dataloaders.dataloader_cifar10 import get_dataloaders
    trainloader, testloader = get_dataloaders(config)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config["lr"])
    criterion = nn.MSELoss()

    # ---- Load classifier ----
    classifier = SimpleCNN().to(device)
    classifier.load_state_dict(torch.load(
        os.path.join(main_path, "train_cifar10_classification/experiments/exp_20260505_193205/best_model.pth")
    ))
    classifier.eval()

    # ---- Logging ----
    suffix = "fixed_snr_20" if mode == "fixed" else "dynamic_snr_10to30"
    csv_file = os.path.join(main_path, f"train_log_{suffix}.csv")

    start_time = time.time()

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        encoder.train()
        decoder.train()

        train_loss, train_acc, train_psnr = 0, 0, 0

        for x, _ in trainloader:
            x = x.to(device)

            if mode == "fixed":
                snr = 20
            else:
                snr = torch.randint(10, 31, (1,)).item()

            z = encoder(x)
            z_noisy = add_awgn_noise(z, snr)
            x_hat = decoder(z_noisy)

            loss = criterion(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_psnr += compute_psnr(x, x_hat).item()
            train_acc += compute_accuracy(classifier, x, x_hat)

        # ---- TEST ----
        encoder.eval()
        decoder.eval()

        test_loss, test_acc, test_psnr = 0, 0, 0

        with torch.no_grad():
            for x, _ in testloader:
                x = x.to(device)

                if mode == "fixed":
                    snr = 20
                else:
                    snr = torch.randint(10, 31, (1,)).item()

                z = encoder(x)
                z_noisy = add_awgn_noise(z, snr)
                x_hat = decoder(z_noisy)

                loss = criterion(x_hat, x)

                test_loss += loss.item()
                test_psnr += compute_psnr(x, x_hat).item()
                test_acc += compute_accuracy(classifier, x, x_hat)

        # ---- AVERAGE ----
        train_loss /= len(trainloader)
        train_acc /= len(trainloader)
        train_psnr /= len(trainloader)

        test_loss /= len(testloader)
        test_acc /= len(testloader)
        test_psnr /= len(testloader)

        epoch_time = time.time() - epoch_start
        est_total = estimate_total_time(start_time, epoch, config["epochs"])

        log_csv(csv_file, [
            epoch,
            train_acc,
            train_loss,
            train_psnr,
            test_acc,
            test_loss,
            test_psnr,
            epoch_time,
            est_total
        ])

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")

    # ---- SAVE MODELS ----
    torch.save(encoder.state_dict(),
               os.path.join(main_path, f"encoder_{suffix}.pth"))

    torch.save(decoder.state_dict(),
               os.path.join(main_path, f"decoder_{suffix}.pth"))