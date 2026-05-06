import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=config["dataset_path"], train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=config["dataset_path"], train=False, download=True, transform=transform
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