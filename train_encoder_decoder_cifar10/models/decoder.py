import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self, compression_rate=1.0):
        super().__init__()

        self.dimension = int(96 * 96 * 3 * compression_rate / (8 * 8))

        self.tconv1 = nn.ConvTranspose2d(
            self.dimension,
            512,
            kernel_size=3,
            stride=1,
            padding=0
        )

        self.tconv2 = nn.ConvTranspose2d(
            512,
            384,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.tconv3 = nn.ConvTranspose2d(
            384,
            256,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.tconv4 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.tconv5 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=5,
            stride=2,
            padding=0
        )

        self.tconv6 = nn.ConvTranspose2d(
            64,
            3,
            kernel_size=4,
            stride=2,
            padding=0
        )

    def forward(self, x):

        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))
        x = F.relu(self.tconv5(x))
        x = torch.tanh(self.tconv6(x))

        return x