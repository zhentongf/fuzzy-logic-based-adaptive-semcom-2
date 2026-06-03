import os

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


def get_dataloaders(config):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,),
            (0.3081,)
        )
    ])

    train_dataset = datasets.MNIST(
        root=config["dataset_path"],
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=config["dataset_path"],
        train=False,
        download=False,
        transform=transform
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    return trainloader, testloader