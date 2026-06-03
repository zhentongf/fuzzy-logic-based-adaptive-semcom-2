import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Encoder_Decoder(nn.Module):

    def __init__(self, channel=128):
        super().__init__()

        self.fc1_1 = nn.Linear(
            28 * 28,
            1024
        )

        self.fc1_2 = nn.Linear(
            1024,
            channel
        )

        self.fc2_1 = nn.Linear(
            channel,
            1024
        )

        self.fc2_2 = nn.Linear(
            1024,
            28 * 28
        )

    def encode(self, x):

        x = x.view(
            x.size(0),
            -1
        )

        x = F.relu(
            self.fc1_1(x)
        )

        x = F.relu(
            self.fc1_2(x)
        )

        return x

    def decode(self, x):

        x = F.relu(
            self.fc2_1(x)
        )

        x = torch.sigmoid(
            self.fc2_2(x)
        )

        x = x.view(
            -1,
            1,
            28,
            28
        )

        return x

    def forward(self, x):

        latent = self.encode(x)

        reconstructed = self.decode(
            latent
        )

        return reconstructed