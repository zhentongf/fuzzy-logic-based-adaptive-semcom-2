import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, compression_rate=1.0):
        super().__init__()

        self.dimension = int(96 * 96 * 3 * compression_rate / (8 * 8))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(384, 512, kernel_size=5, stride=1, padding=0)
        self.conv6 = nn.Conv2d(512, self.dimension, kernel_size=3, stride=1, padding=0)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        return x