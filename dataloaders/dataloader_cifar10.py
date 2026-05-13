import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(config):

    transform_train = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    transform_test = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=config["dataset_path"],
        train=True,
        download=False,
        transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root=config["dataset_path"],
        train=False,
        download=False,
        transform=transform_test
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