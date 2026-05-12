import torch
import torch.nn as nn


def conv_relu(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.ReLU(True)
    )
    return layer


class Inception(nn.Module):
    def __init__(
        self,
        in_channel,
        out1_1,
        out2_1,
        out2_3,
        out3_1,
        out3_5,
        out4_1
    ):
        super(Inception, self).__init__()

        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)

        return torch.cat((f1, f2, f3, f4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, verbose=False):
        super(GoogLeNet, self).__init__()

        self.verbose = verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, 64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )

        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )

        self.block3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )

        self.block4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )

        self.block5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 182, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):

        x = self.block1(x)

        if self.verbose:
            print("block1:", x.shape)

        x = self.block2(x)

        if self.verbose:
            print("block2:", x.shape)

        x = self.block3(x)

        if self.verbose:
            print("block3:", x.shape)

        x = self.block4(x)

        if self.verbose:
            print("block4:", x.shape)

        x = self.block5(x)

        if self.verbose:
            print("block5:", x.shape)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x