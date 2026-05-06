import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 8x8
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )

    def forward(self, x):
        return self.net(x)