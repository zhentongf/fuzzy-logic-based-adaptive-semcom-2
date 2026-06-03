import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

from dataloaders.dataloader_mnist import (
    get_dataloaders
)

from train_encoder_decoder_mnist.models.mlp_encoder_decoder import (
    MLP_Encoder_Decoder
)

from train_encoder_decoder_mnist.models.channel import (
    add_awgn_noise
)

from train_encoder_decoder_mnist.utils.metrics import (
    compute_psnr,
    compute_accuracy
)

from train_encoder_decoder_mnist.utils.logger import (
    log_csv
)

from train_mnist_classification.models.mlp_classifier import (
    MLP_Classifier
)


def get_config(main_path):

    config = {

        "dataset_path": os.path.join(
            main_path,
            "datasets",
            "MNIST"
        ),

        "batch_size": 64,

        "num_workers": 0,

        "epochs": 20,

        "lr": 1e-3,

        "channel": 128,

        "classifier_path": os.path.join(
            main_path,
            "train_mnist_classification",
            "experiments",
            "exp_20260603_074954",
            "best_model.pth"
        ),

        "experiments_path": os.path.join(
            main_path,
            "train_encoder_decoder_mnist",
            "experiments"
        )
    }

    return config


def get_snr(mode):

    if mode == "fixed":
        return 20

    return random.randint(
        10,
        30
    )


def train(
    main_path,
    mode="fixed"
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = get_config(
        main_path
    )

    trainloader, testloader = (
        get_dataloaders(config)
    )

    model = MLP_Encoder_Decoder(
        channel=config["channel"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    criterion = nn.MSELoss()

    # --------------------------------------------------
    # Load MNIST classifier
    # --------------------------------------------------

    classifier = MLP_Classifier().to(device)

    classifier.load_state_dict(
        torch.load(
            config["classifier_path"],
            map_location=device
        )
    )

    classifier.eval()

    for param in classifier.parameters():
        param.requires_grad = False

    # --------------------------------------------------
    # Experiment directory
    # --------------------------------------------------

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    if mode == "fixed":
        mode_name = "fixed_snr_20"
    else:
        mode_name = (
            "dynamic_snr_10to30"
        )

    exp_name = (
        f"exp_{timestamp}_{mode_name}"
    )

    exp_dir = os.path.join(
        config["experiments_path"],
        exp_name
    )

    os.makedirs(
        exp_dir,
        exist_ok=True
    )

    csv_file = os.path.join(
        exp_dir,
        "training_log.csv"
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(
        config["epochs"]
    ):

        epoch_start = time.time()

        model.train()

        train_loss = 0
        train_psnr = 0
        train_acc = 0

        # ----------------------------------------------
        # TRAIN
        # ----------------------------------------------

        for images, _ in trainloader:

            images = images.to(
                device
            )

            snr = get_snr(mode)

            encoded = model.encode(
                images
            )

            noisy_encoded = (
                add_awgn_noise(
                    encoded,
                    snr
                )
            )

            reconstructed = (
                model.decode(
                    noisy_encoded
                )
            )

            loss = criterion(
                reconstructed,
                images
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
            )

            train_psnr += (
                compute_psnr(
                    images,
                    reconstructed
                ).item()
            )

            train_acc += (
                compute_accuracy(
                    classifier,
                    images,
                    reconstructed
                )
            )

        # ----------------------------------------------
        # TEST
        # ----------------------------------------------

        model.eval()

        test_loss = 0
        test_psnr = 0
        test_acc = 0

        with torch.no_grad():

            for images, _ in testloader:

                images = images.to(
                    device
                )

                snr = get_snr(mode)

                encoded = model.encode(
                    images
                )

                noisy_encoded = (
                    add_awgn_noise(
                        encoded,
                        snr
                    )
                )

                reconstructed = (
                    model.decode(
                        noisy_encoded
                    )
                )

                loss = criterion(
                    reconstructed,
                    images
                )

                test_loss += (
                    loss.item()
                )

                test_psnr += (
                    compute_psnr(
                        images,
                        reconstructed
                    ).item()
                )

                test_acc += (
                    compute_accuracy(
                        classifier,
                        images,
                        reconstructed
                    )
                )

        # ----------------------------------------------
        # Average metrics
        # ----------------------------------------------

        train_loss /= len(
            trainloader
        )

        train_psnr /= len(
            trainloader
        )

        train_acc /= len(
            trainloader
        )

        test_loss /= len(
            testloader
        )

        test_psnr /= len(
            testloader
        )

        test_acc /= len(
            testloader
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        log_csv(
            csv_file,
            [
                epoch + 1,
                train_acc,
                train_loss,
                train_psnr,
                test_acc,
                test_loss,
                test_psnr,
                epoch_time
            ]
        )

        print(
            f"Epoch [{epoch+1}/{config['epochs']}] "
            f"SNR={snr} "
            f"Train Loss={train_loss:.6f} "
            f"Test Loss={test_loss:.6f} "
            f"Train PSNR={train_psnr:.2f} "
            f"Test PSNR={test_psnr:.2f}"
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                exp_dir,
                "mlp_latest.pth"
            )
        )

    torch.save(
        model.state_dict(),
        os.path.join(
            exp_dir,
            "mlp_final.pth"
        )
    )

    print(
        "\nTraining completed."
    )

    print(
        f"Experiment saved to:\n{exp_dir}"
    )