import os
import graphs_scripts.train_encoder_decoder_cifar10_graph as train_encoder_decoder_cifar10_graph
import graphs_scripts.transmit_cifar10_composite_snr_graph as transmit_cifar10_composite_snr_graph
import graphs_scripts.transmit_cifar10_graph as transmit_cifar10_graph

import graphs_scripts.train_encoder_decoder_mnist_graph as train_encoder_decoder_mnist_graph
import graphs_scripts.transmit_mnist_composite_snr_graph as transmit_mnist_composite_snr_graph
import graphs_scripts.transmit_mnist_graph as transmit_mnist_graph



main_path = os.path.dirname(os.path.abspath(__file__))
def get_main_config():
    config = {
        "train_encoder_decoder_cifar10_graph": False,
        "transmit_cifar10_composite_snr_graph": False,
        "transmit_cifar10_graph": False,
        "train_encoder_decoder_mnist_graph": False,
        "transmit_mnist_composite_snr_graph": True,
        "transmit_mnist_graph": True,
        "main_path": main_path
    }
    return config

def main():
    config = get_main_config()
    
    if config["train_encoder_decoder_cifar10_graph"]:
        print("Starting CIFAR-10 encoder-decoder graph drawing...")
        train_encoder_decoder_cifar10_graph.draw(config["main_path"])
    
    if config["transmit_cifar10_composite_snr_graph"]:
        print("Starting CIFAR-10 composite SNR graph drawing...")
        transmit_cifar10_composite_snr_graph.draw(config["main_path"])
    
    if config["transmit_cifar10_graph"]:
        print("Starting CIFAR-10 transmission graph drawing...")
        transmit_cifar10_graph.draw(config["main_path"])

    if config["train_encoder_decoder_mnist_graph"]:
        print("Starting MNIST encoder-decoder graph drawing...")
        train_encoder_decoder_mnist_graph.draw(config["main_path"])

    if config["transmit_mnist_composite_snr_graph"]:
        print("Starting MNIST composite SNR graph drawing...")
        transmit_mnist_composite_snr_graph.draw(config["main_path"])
    
    if config["transmit_mnist_graph"]:
        print("Starting MNIST transmission graph drawing...")
        transmit_mnist_graph.draw(config["main_path"])
        
if __name__ == "__main__":
    main()