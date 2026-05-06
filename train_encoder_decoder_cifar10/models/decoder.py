import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),   # 32x32
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 8, 8)
        return self.net(x)