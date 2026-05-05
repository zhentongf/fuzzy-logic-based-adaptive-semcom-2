import train_cifar10_classification.main_train_cifar10_classification as train_cifar10_classification

def get_main_config():
    config = {
        "train_cifar10_classification": True
    }
    return config

def main():
    config = get_main_config()

    if config["train_cifar10_classification"]:
        print("Starting CIFAR-10 classification training...")
        train_cifar10_classification.run()


if __name__ == "__main__":
    main()