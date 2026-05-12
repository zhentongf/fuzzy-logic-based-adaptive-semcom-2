import torch
import torch.nn as nn
import torch.optim as optim

import time
import os
import json
import csv

from datetime import datetime

from dataloaders.dataloader_cifar10 import get_dataloaders
from train_cifar10_classification.models.googlenet import GoogLeNet


# =========================
# 1. Config System
# =========================
def get_config(main_path):

    config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 30,
        "num_workers": 0,

        "experiment_root": os.path.join(
            main_path,
            "train_cifar10_classification/experiments"
        ),

        "dataset_path": os.path.join(
            main_path,
            "datasets/CIFAR10"
        ),

        "model_name": "GoogLeNet"
    }

    return config


# =========================
# 2. Experiment Setup
# =========================
def create_experiment_folder(config):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_path = os.path.join(
        config["experiment_root"],
        f"exp_{timestamp}"
    )

    os.makedirs(exp_path, exist_ok=True)

    return exp_path


def save_config(config, path):

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


# =========================
# 3. Train One Epoch
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()

    train_acc = 100.0 * correct / total
    train_loss = running_loss / len(loader)

    return train_loss, train_acc


# =========================
# 4. Evaluation
# =========================
def evaluate(model, loader, criterion, device):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, targets in loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

    test_acc = 100.0 * correct / total
    test_loss = running_loss / len(loader)

    return test_loss, test_acc


# =========================
# 5. CSV Logging
# =========================
def init_csv(path):

    csv_path = os.path.join(path, "log.csv")

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "test_loss",
            "test_acc",
            "epoch_time"
        ])

    return csv_path


def log_csv(csv_file, row):

    with open(csv_file, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(row)


# =========================
# 6. Checkpoint Saving
# =========================
def save_checkpoint(model, path, filename):

    torch.save(
        model.state_dict(),
        os.path.join(path, filename)
    )


# =========================
# 7. Main Runner
# =========================
def run(main_path):

    config = get_config(main_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    exp_path = create_experiment_folder(config)

    save_config(config, exp_path)

    trainloader, testloader = get_dataloaders(config)

    model = GoogLeNet(
        in_channel=3,
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    csv_file = init_csv(exp_path)

    best_acc = 0

    for epoch in range(config["epochs"]):

        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            trainloader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_acc = evaluate(
            model,
            testloader,
            criterion,
            device
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{config['epochs']}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        save_checkpoint(
            model,
            exp_path,
            "last_model.pth"
        )

        if test_acc > best_acc:

            best_acc = test_acc

            save_checkpoint(
                model,
                exp_path,
                "best_model.pth"
            )

            print(f"New Best Accuracy: {best_acc:.2f}%")

        log_csv(csv_file, [
            epoch + 1,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
            epoch_time
        ])

    print(f"\nTraining Finished.")
    print(f"Best Test Accuracy: {best_acc:.2f}%")