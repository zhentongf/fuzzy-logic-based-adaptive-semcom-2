import os
import train_cifar10_classification.main_train_cifar10_classification as train_cifar10_classification

main_path = os.path.dirname(os.path.abspath(__file__))
def get_main_config():
    config = {
        "train_cifar10_classification": False,
        "main_path": main_path
    }
    return config

def main():
    config = get_main_config()

    if config["train_cifar10_classification"]:
        print("Starting CIFAR-10 classification training...")
        train_cifar10_classification.run(config["main_path"])


if __name__ == "__main__":
    main()