import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import joblib
import os

from reward_prediction_network.model import RewardPredictionNetwork

def train(main_path):
    # Configuration
    CSV_FILES = [
        os.path.join(
            main_path,
            "transmit_cifar10/experiments/transmission_results_20260701_021626_noise_0.6.csv"
        ),
        os.path.join(
            main_path,
            "transmit_mnist/experiments/transmission_results_20260701_023433_noise_0.6.csv"
        )
    ]

    PSNR_WEIGHT = 0.7
    ACC_WEIGHT = 0.3

    EPOCHS = 500
    LEARNING_RATE = 1e-3
    
    # Load and combine all CSV files
    df_list = []

    for csv_file in CSV_FILES:
        temp_df = pd.read_csv(csv_file)

        # Optional: record which dataset each row comes from
        temp_df["dataset"] = os.path.basename(csv_file)

        df_list.append(temp_df)

    df = pd.concat(
        df_list,
        ignore_index=True
    )

    print(f"Total samples: {len(df)}")

    # Separate Cases
    semantic_df = df[
        df["testing_case"] == "all_dynamic_snr"
    ].copy()

    direct_df = df[
        df["testing_case"] == "all_direct"
    ].copy()

    # Match Corresponding Rows
    key_cols = [
        "time",
        "car_ID_TX",
        "car_ID_RX",
        "snr_values",
        "distance_values",
        "rel_speed_values"
    ]

    merged = semantic_df.merge(
        direct_df,
        on=key_cols,
        suffixes=("_semantic", "_direct")
    )

    # Compute Gains
    merged["psnr_gain"] = (
        merged["psnr_semantic"]
        - merged["psnr_direct"]
    )

    merged["acc_gain"] = (
        merged["accuracy_semantic"]
        - merged["accuracy_direct"]
    )

    # Robust Normalization Function
    def robust_symmetric_normalize(values):

        values = np.asarray(values)

        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)

        trimmed = values[
            (values >= lower) &
            (values <= upper)
        ]

        base = np.max(np.abs(trimmed))

        normalized = values / base

        normalized = np.clip(
            normalized,
            -1.0,
            1.0
        )

        return normalized, base

    # Normalize PSNR Gain
    merged["psnr_gain_norm"], psnr_base = (
        robust_symmetric_normalize(
            merged["psnr_gain"]
        )
    )

    # Normalize Accuracy Gain
    merged["acc_gain_norm"], acc_base = (
        robust_symmetric_normalize(
            merged["acc_gain"]
        )
    )

    # Save Bases
    joblib.dump(
        {
            "psnr_base": psnr_base,
            "acc_base": acc_base
        },
        os.path.join(main_path, "reward_prediction_network/noise_0.6/reward_normalization.pkl")
    )

    # Build Reward
    merged["reward"] = (
        PSNR_WEIGHT *
        merged["psnr_gain_norm"]
        +
        ACC_WEIGHT *
        merged["acc_gain_norm"]
    )


    # Prepare Input Features
    X = merged[
        [
            "snr_values",
            "distance_values",
            "rel_speed_values"
        ]
    ].values

    y = merged["reward"].values

    # Normalize Inputs
    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)

    joblib.dump(
        scaler,
        os.path.join(main_path, "reward_prediction_network/noise_0.6/state_scaler.pkl")
    )

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Convert to Torch
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    y_train = torch.FloatTensor(
        y_train.reshape(-1, 1)
    )

    y_test = torch.FloatTensor(
        y_test.reshape(-1, 1)
    )

    # Create Model
    model = RewardPredictionNetwork()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Training
    for epoch in range(EPOCHS):

        model.train()

        pred = model(X_train)

        loss = criterion(
            pred,
            y_train
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 50 == 0:

            print(
                f"Epoch {epoch:4d} "
                f"Loss = {loss.item():.6f}"
            )

    # Evaluation
    model.eval()

    with torch.no_grad():

        pred_test = model(X_test)

    mse = mean_squared_error(
        y_test.numpy(),
        pred_test.numpy()
    )

    print()
    print("Test MSE:", mse)

    # Save Model
    torch.save(
        model.state_dict(),
        os.path.join(main_path, "reward_prediction_network/noise_0.6/reward_prediction_network.pth")
    )

    print("Model saved.")


