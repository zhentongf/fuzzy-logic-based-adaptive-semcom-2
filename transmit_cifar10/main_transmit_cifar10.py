import os
import random
import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from dataloaders.dataloader_cifar10 import get_dataloaders

from train_cifar10_classification.models.googlenet import GoogLeNet

from train_encoder_decoder_cifar10.models.encoder import Encoder
from train_encoder_decoder_cifar10.models.decoder import Decoder
from train_encoder_decoder_cifar10.models.channel import add_awgn_noise

from train_encoder_decoder_cifar10.utils.metrics import (
    compute_psnr,
    compute_accuracy
)


def get_config(main_path):

    config = {
        "main_path": main_path,

        # dataset
        "dataset_path": os.path.join(main_path, "datasets/CIFAR10"),
        "batch_size": 64,
        "num_workers": 0,

        # csv
        "nearest_cars_csv": os.path.join(
            main_path,
            "nearest_cars_data.csv"
        ),

        # classifier
        "classifier_model_path": os.path.join(
            main_path,
            "train_cifar10_classification",
            "experiments",
            "exp_20260512_144813",
            "best_model.pth"
        ),

        # fixed snr semantic model
        "fixed_encoder_path": os.path.join(
            main_path,
            "train_encoder_decoder_cifar10",
            "experiments",
            "exp_20260512_184610_fixed_snr_20",
            "encoder_final.pth"
        ),

        "fixed_decoder_path": os.path.join(
            main_path,
            "train_encoder_decoder_cifar10",
            "experiments",
            "exp_20260512_184610_fixed_snr_20",
            "decoder_final.pth"
        ),

        # dynamic snr semantic model
        "dynamic_encoder_path": os.path.join(
            main_path,
            "train_encoder_decoder_cifar10",
            "experiments",
            "exp_20260512_202705_dynamic_snr_10to30",
            "encoder_final.pth"
        ),

        "dynamic_decoder_path": os.path.join(
            main_path,
            "train_encoder_decoder_cifar10",
            "experiments",
            "exp_20260512_202705_dynamic_snr_10to30",
            "decoder_final.pth"
        ),

        # experiment output
        "experiment_dir": os.path.join(
            main_path,
            "transmit_cifar10",
            "experiments"
        ),

        # transmission
        "compression_rate": 1.0,
        "num_test_images": 1000
    }

    return config


def load_classifier(config, device):

    model = GoogLeNet(
        in_channel=3,
        num_classes=10
    )

    state_dict = torch.load(
        config["classifier_model_path"],
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def load_encoder_decoder(
    encoder_path,
    decoder_path,
    compression_rate,
    device
):

    encoder = Encoder(compression_rate=compression_rate)
    decoder = Decoder(compression_rate=compression_rate)

    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device)
    )

    decoder.load_state_dict(
        torch.load(decoder_path, map_location=device)
    )

    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()

    return encoder, decoder


def get_test_loader(config):

    _, testloader_full = get_dataloaders(config)

    test_dataset = testloader_full.dataset

    random.seed(42)

    indices = list(range(len(test_dataset)))
    random.shuffle(indices)

    selected_indices = indices[:config["num_test_images"]]

    subset_dataset = Subset(
        test_dataset,
        selected_indices
    )

    subset_loader = DataLoader(
        subset_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    return subset_loader


def semantic_transmission(
    images,
    encoder,
    decoder,
    snr_db
):
    """
    Semantic transmission:
    image -> encoder -> AWGN noise -> decoder
    """

    latent = encoder(images)

    noisy_latent = add_awgn_noise(
        latent,
        snr_db
    )

    reconstructed = decoder(noisy_latent)

    return reconstructed


def direct_transmission(
    images,
    snr_db
):
    """
    Direct transmission:
    image -> AWGN noise
    """

    noisy_images = add_awgn_noise(
        images,
        snr_db
    )

    return noisy_images


def evaluate_transmission(
    dataloader,
    classifier,
    mode,
    composite_snr_db,
    use_nn,
    fixed_encoder,
    fixed_decoder,
    dynamic_encoder,
    dynamic_decoder,
    device
):

    total_psnr = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            # ==================================================
            # CASE 1
            # fuzzy logic + fixed snr semantic model
            # ==================================================
            if mode == "fuzzy_logic_fixed_snr":

                if use_nn:
                    outputs = semantic_transmission(
                        images,
                        fixed_encoder,
                        fixed_decoder,
                        composite_snr_db
                    )
                else:
                    outputs = direct_transmission(
                        images,
                        composite_snr_db
                    )

            # ==================================================
            # CASE 2
            # fuzzy logic + dynamic snr semantic model
            # ==================================================
            elif mode == "fuzzy_logic_dynamic_snr":

                if use_nn:
                    outputs = semantic_transmission(
                        images,
                        dynamic_encoder,
                        dynamic_decoder,
                        composite_snr_db
                    )
                else:
                    outputs = direct_transmission(
                        images,
                        composite_snr_db
                    )

            # ==================================================
            # CASE 3
            # all semantic dynamic snr
            # ==================================================
            elif mode == "all_dynamic_snr":

                outputs = semantic_transmission(
                    images,
                    dynamic_encoder,
                    dynamic_decoder,
                    composite_snr_db
                )

            # ==================================================
            # CASE 4
            # all direct
            # ==================================================
            elif mode == "all_direct":

                outputs = direct_transmission(
                    images,
                    composite_snr_db
                )

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # ----------------------------------------
            # metrics
            # ----------------------------------------

            psnr = compute_psnr(
                images,
                outputs
            )

            acc = compute_accuracy(
                classifier,
                images,
                outputs
            )

            total_psnr += psnr.item()
            total_acc += acc
            total_batches += 1

    avg_psnr = total_psnr / total_batches
    avg_acc = total_acc / total_batches

    return avg_psnr, avg_acc


def run(main_path):

    config = get_config(main_path)

    os.makedirs(
        config["experiment_dir"],
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ======================================================
    # load classifier
    # ======================================================

    print("Loading GoogLeNet classifier...")

    classifier = load_classifier(
        config,
        device
    )

    # ======================================================
    # load semantic models
    # ======================================================

    print("Loading fixed SNR semantic model...")

    fixed_encoder, fixed_decoder = load_encoder_decoder(
        config["fixed_encoder_path"],
        config["fixed_decoder_path"],
        config["compression_rate"],
        device
    )

    print("Loading dynamic SNR semantic model...")

    dynamic_encoder, dynamic_decoder = load_encoder_decoder(
        config["dynamic_encoder_path"],
        config["dynamic_decoder_path"],
        config["compression_rate"],
        device
    )

    # ======================================================
    # load CIFAR10 test subset
    # ======================================================

    print("Loading CIFAR10 test subset...")

    testloader = get_test_loader(config)

    # ======================================================
    # load nearest cars data
    # ======================================================

    print("Loading nearest_cars_data.csv ...")

    df = pd.read_csv(
        config["nearest_cars_csv"]
    )

    # ======================================================
    # filter distance <= 200m
    # ======================================================

    df = df[
        df["distance_values"] <= 200
    ].reset_index(drop=True)

    print(f"Filtered rows: {len(df)}")

    # ======================================================
    # testing cases
    # ======================================================

    testing_cases = [
        "fuzzy_logic_fixed_snr",
        "fuzzy_logic_dynamic_snr",
        "all_dynamic_snr",
        "all_direct"
    ]

    results = []

    # ======================================================
    # main loop
    # ======================================================

    for testing_case in testing_cases:

        print("\n=================================================")
        print(f"Testing case: {testing_case}")
        print("=================================================")

        for idx, row in df.iterrows():

            composite_snr_db = float(
                row["composite_snr_db"]
            )

            use_nn = bool(
                row["use_nn"]
            )

            avg_psnr, avg_acc = evaluate_transmission(
                dataloader=testloader,
                classifier=classifier,
                mode=testing_case,
                composite_snr_db=composite_snr_db,
                use_nn=use_nn,
                fixed_encoder=fixed_encoder,
                fixed_decoder=fixed_decoder,
                dynamic_encoder=dynamic_encoder,
                dynamic_decoder=dynamic_decoder,
                device=device
            )

            result_row = row.to_dict()

            result_row["testing_case"] = testing_case
            result_row["psnr"] = round(avg_psnr, 4)
            result_row["accuracy"] = round(avg_acc, 4)

            results.append(result_row)

            print(
                f"[{testing_case}] "
                f"Row {idx + 1}/{len(df)} | "
                f"SNR={composite_snr_db:.2f} | "
                f"use_nn={use_nn} | "
                f"PSNR={avg_psnr:.4f} | "
                f"ACC={avg_acc:.4f}"
            )

    # ======================================================
    # save results
    # ======================================================

    results_df = pd.DataFrame(results)

    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_csv = os.path.join(
        config["experiment_dir"],
        f"transmission_results_{timestamp}.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False
    )

    print("\n=================================================")
    print("Experiment completed!")
    print(f"Saved results to:\n{output_csv}")
    print("=================================================")

