import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import time
import os
import json
import csv
from datetime import datetime

# =========================
# 1. Config System
# =========================
def get_config():
    config = {
        "batch_size": 128,
        "learning_rate": 0.001,
        "epochs": 20,
        "num_workers": 2,
        "experiment_root": "./experiments",
        "model_name": "SimpleCNN"
    }
    return config


# =========================
# 2. Experiment Setup
# =========================
def create_experiment_folder(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_path = os.path.join(config["experiment_root"], f"exp_{timestamp}")
    os.makedirs(exp_path, exist_ok=True)
    return exp_path


def save_config(config, path):
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


# =========================
# 3. Dataset
# =========================
def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="../datasets/CIFAR10", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="../datasets/CIFAR10", train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    return trainloader, testloader


# =========================
# 4. Model
# =========================
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


# =========================
# 5. Train & Eval
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct, total = 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()

    return 100. * correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()

    return 100. * correct / total


# =========================
# 6. Logging
# =========================
def init_csv(path):
    file = os.path.join(path, "log.csv")
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_acc", "test_acc",
            "epoch_time", "estimated_total"
        ])
    return file


def log_csv(file, row):
    with open(file, "a", newline="") as f:
        csv.writer(f).writerow(row)


# =========================
# 7. Checkpointing
# =========================
def save_checkpoint(model, path, name):
    torch.save(model.state_dict(), os.path.join(path, name))


# =========================
# 8. Main Experiment Runner
# =========================
def run():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    exp_path = create_experiment_folder(config)
    save_config(config, exp_path)

    trainloader, testloader = get_dataloaders(config)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    csv_file = init_csv(exp_path)

    best_acc = 0
    epoch_times = []

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()

        train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        test_acc = evaluate(model, testloader, device)

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

        avg_time = sum(epoch_times) / len(epoch_times)
        est_total = avg_time * config["epochs"]

        print(f"[{epoch}/{config['epochs']}] "
              f"Train: {train_acc:.2f} | Test: {test_acc:.2f} | "
              f"Time: {epoch_time:.2f}s | Est Total: {est_total:.2f}s")

        # Save last checkpoint
        save_checkpoint(model, exp_path, "last_model.pth")

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, exp_path, "best_model.pth")

        # Log CSV
        log_csv(csv_file, [
            epoch, train_acc, test_acc,
            epoch_time, est_total
        ])

    print(f"\nBest Test Accuracy: {best_acc:.2f}%")


# if __name__ == "__main__":
#     run()