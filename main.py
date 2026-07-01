import os
import train_cifar10_classification.main_train_cifar10_classification as train_cifar10_classification
import train_encoder_decoder_cifar10.main_train_encoder_decoder_cifar10 as train_encoder_decoder_cifar10
import transmit_cifar10.main_transmit_cifar10 as transmit_cifar10

import train_mnist_classification.main_train_mnist_classification as train_mnist_classification
import train_encoder_decoder_mnist.main_train_encoder_decoder_mnist as train_encoder_decoder_mnist
import transmit_mnist.main_transmit_mnist as transmit_mnist

import reward_prediction_network.main_train_reward_prediction_network as train_reward_prediction_network


main_path = os.path.dirname(os.path.abspath(__file__))
def get_main_config():
    config = {
        "train_cifar10_classification": False,
        "train_encoder_decoder_cifar10": False,
        "transmit_cifar10": False,
        "train_mnist_classification": False,
        "train_encoder_decoder_mnist": False,
        "transmit_mnist": False,
        "train_reward_prediction_network": True,
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
        # fixed SNR
        train_encoder_decoder_cifar10.train(
            config["main_path"],
            mode="fixed",
            compression_rate=1.0
        )
        # dynamic SNR
        train_encoder_decoder_cifar10.train(
            config["main_path"],
            mode="dynamic",
            compression_rate=1.0
        )

    if config["transmit_cifar10"]:
        print("Starting CIFAR-10 transmission...")
        transmit_cifar10.run(config["main_path"])

    if config["train_mnist_classification"]:
        print("Starting MNIST classification training...")
        train_mnist_classification.run(config["main_path"])

    if config["train_encoder_decoder_mnist"]:
        print("Starting MNIST encoder-decoder training...")
        train_encoder_decoder_mnist.train(
            config["main_path"],
            mode="fixed"
        )
        train_encoder_decoder_mnist.train(
            config["main_path"],
            mode="dynamic"
       )
    if config["transmit_mnist"]:
        print("Starting MNIST transmission...")
        transmit_mnist.run(config["main_path"])

    if config["train_reward_prediction_network"]:
        print("Starting reward prediction network training...")
        train_reward_prediction_network.train(config["main_path"])



if __name__ == "__main__":
    main()