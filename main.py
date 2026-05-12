import os
import train_cifar10_classification.main_train_cifar10_classification as train_cifar10_classification
import train_encoder_decoder_cifar10.main_train_encoder_decoder_cifar10 as train_encoder_decoder_cifar10

main_path = os.path.dirname(os.path.abspath(__file__))
def get_main_config():
    config = {
        "train_cifar10_classification": True,
        "train_encoder_decoder_cifar10": False,
        "main_path": main_path
    }
    return config

def main():
    config = get_main_config()

    if config["train_cifar10_classification"]:
        print("Starting CIFAR-10 classification training...")
        train_cifar10_classification.run(config["main_path"])

    if config["train_encoder_decoder_cifar10"]:
        print("Starting CIFAR-10 encoder-decoder training...")
        train_encoder_decoder_cifar10.train(config["main_path"], mode="fixed")

if __name__ == "__main__":
    main()