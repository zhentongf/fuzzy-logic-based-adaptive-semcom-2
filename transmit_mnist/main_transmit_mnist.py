import os
import random
import datetime
import pandas as pd
import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader, Subset

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

from train_mnist_classification.models.mlp_classifier import (
    MLP_Classifier
)

from reward_prediction_network.model import RewardPredictionNetwork


def get_config(main_path):

    config = {

        "main_path": main_path,

        # dataset
        "dataset_path": os.path.join(
            main_path,
            "datasets",
            "MNIST"
        ),

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
            "train_mnist_classification",
            "experiments",
            "exp_20260603_074954",
            "best_model.pth"
        ),

        # fixed snr semantic model
        "fixed_model_path": os.path.join(
            main_path,
            "train_encoder_decoder_mnist",
            "experiments",
            "exp_20260603_084727_fixed_snr_20",
            "mlp_final.pth"
        ),

        # dynamic snr semantic model
        "dynamic_model_path": os.path.join(
            main_path,
            "train_encoder_decoder_mnist",
            "experiments",
            "exp_20260603_085208_dynamic_snr_10to30",
            "mlp_final.pth"
        ),

        # reward prediction network
        "reward_model_path": os.path.join(
            main_path,
            "reward_prediction_network",
            "noise_0.8",
            "reward_prediction_network.pth"
        ),

        "state_scaler_path": os.path.join(
            main_path,
            "reward_prediction_network",
            "noise_0.8",
            "state_scaler.pkl"
        ),

        # experiment output
        "experiment_dir": os.path.join(
            main_path,
            "transmit_mnist",
            "experiments"
        ),

        "compression_rate": 1.0,
        "num_test_images": 1000
    }

    return config


def load_classifier(config, device):

    model = MLP_Classifier()

    state_dict = torch.load(
        config["classifier_model_path"],
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def load_semantic_model(
    model_path,
    compression_rate,
    device
):

    model = MLP_Encoder_Decoder(
        channel=int(compression_rate * 128)
    )

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


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
    model,
    snr_db
):
    """
    image -> encoder -> AWGN -> decoder
    """

    latent = model.encode(
        images
    )

    noisy_latent = add_awgn_noise(
        latent,
        snr_db
    )

    reconstructed = model.decode(
        noisy_latent
    )

    return reconstructed


def direct_transmission(
    images,
    snr_db
):

    noisy_images = add_awgn_noise(
        images,
        snr_db
    )

    return noisy_images


def predict_reward(
    reward_model,
    scaler,
    snr,
    distance,
    rel_speed,
    device
):
    state = np.array([[snr, distance, rel_speed]])
    state = scaler.transform(state)
    state = torch.FloatTensor(state).to(device)

    with torch.no_grad():
        reward = reward_model(state)

    return reward.item()


def evaluate_transmission(
    dataloader,
    classifier,
    mode,
    composite_snr_db,
    use_nn,
    fixed_model,
    dynamic_model,
    reward_model,
    state_scaler,
    snr,
    distance,
    rel_speed,
    device
):

    total_psnr = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            # ==========================================
            # CASE 1
            # ==========================================

            if mode == "fuzzy_logic_fixed_snr":

                if use_nn:

                    outputs = semantic_transmission(
                        images,
                        fixed_model,
                        composite_snr_db
                    )

                else:

                    outputs = direct_transmission(
                        images,
                        composite_snr_db
                    )

            # ==========================================
            # CASE 2
            # ==========================================

            elif mode == "fuzzy_logic_dynamic_snr":

                if use_nn:

                    outputs = semantic_transmission(
                        images,
                        dynamic_model,
                        composite_snr_db
                    )

                else:

                    outputs = direct_transmission(
                        images,
                        composite_snr_db
                    )

            # ==========================================
            # CASE 3
            # ==========================================

            elif mode == "all_dynamic_snr":

                outputs = semantic_transmission(
                    images,
                    dynamic_model,
                    composite_snr_db
                )

            # ==========================================
            # CASE 4
            # ==========================================

            elif mode == "all_direct":

                outputs = direct_transmission(
                    images,
                    composite_snr_db
                )

            # ==========================================
            # CASE 5
            # deep learning adaptive
            # ==========================================
            elif mode == "deep_learning_adaptive":
                reward = predict_reward(
                    reward_model,
                    state_scaler,
                    snr,
                    distance,
                    rel_speed,
                    device
                )
                if reward > 0:
                    outputs = semantic_transmission(
                        images,
                        dynamic_model,
                        composite_snr_db
                    )
                else:
                    outputs = direct_transmission(
                        images,
                        composite_snr_db
                    )

            else:

                raise ValueError(
                    f"Unknown mode: {mode}"
                )

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

    # ==================================================
    # classifier
    # ==================================================

    print("Loading MNIST classifier...")

    classifier = load_classifier(
        config,
        device
    )

    # ==================================================
    # semantic models
    # ==================================================

    print("Loading fixed SNR model...")

    fixed_model = load_semantic_model(
        config["fixed_model_path"],
        config["compression_rate"],
        device
    )

    print("Loading dynamic SNR model...")

    dynamic_model = load_semantic_model(
        config["dynamic_model_path"],
        config["compression_rate"],
        device
    )

    print("Loading reward prediction network...")

    reward_model = RewardPredictionNetwork()
    reward_model.load_state_dict(
        torch.load(
            config["reward_model_path"],
            map_location=device
        )
    )
    reward_model.eval()
    reward_model.to(device)

    state_scaler = joblib.load(
        config["state_scaler_path"]
    )

    # ==================================================
    # test subset
    # ==================================================

    print("Loading MNIST test subset...")

    testloader = get_test_loader(config)

    # ==================================================
    # nearest cars data
    # ==================================================

    print("Loading nearest_cars_data.csv ...")

    df = pd.read_csv(
        config["nearest_cars_csv"]
    )

    df = df[
        df["distance_values"] <= 200
    ].reset_index(drop=True)

    print(
        f"Filtered rows: {len(df)}"
    )

    testing_cases = [
        "fuzzy_logic_fixed_snr",
        "fuzzy_logic_dynamic_snr",
        "all_dynamic_snr",
        "all_direct",
        "deep_learning_adaptive"
    ]

    results = []

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

            snr = float(
                row["snr_values"]
            )

            distance = float(
                row["distance_values"]
            )

            rel_speed = float(
                row["rel_speed_values"]
            )

            avg_psnr, avg_acc = evaluate_transmission(
                dataloader=testloader,
                classifier=classifier,
                mode=testing_case,
                composite_snr_db=composite_snr_db,
                use_nn=use_nn,
                fixed_model=fixed_model,
                dynamic_model=dynamic_model,
                reward_model=reward_model,
                state_scaler=state_scaler,
                snr=snr,
                distance=distance,
                rel_speed=rel_speed,
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

