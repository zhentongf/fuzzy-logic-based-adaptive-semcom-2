import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Classifier(nn.Module):
    """
    MNIST Digit Classifier
    """

    def __init__(self):

        super(MLP_Classifier, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 125)
        self.fc4 = nn.Linear(125, 10)

    def forward(self, x):

        x = x.view(-1, 28 * 28)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return x