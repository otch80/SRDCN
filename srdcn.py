import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import torch
from torch import nn
from glob import glob
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class SRCNN(nn.Module):
    def __init__(self, in_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    # Activation
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out


class SRDL(nn.Module):  # SR-based Dense Layer model
    def __init__(self):
        super(SRDL, self).__init__()

        n_channels = 3

        self.conv1 = SRCNN(n_channels)
        self.conv2 = SRCNN(n_channels * 2)
        self.conv3 = SRCNN(n_channels * 3)
        self.conv4 = SRCNN(n_channels * 4)
        self.conv5 = SRCNN(n_channels * 5)
        self.conv6 = SRCNN(n_channels * 6)
        self.conv7 = SRCNN(n_channels * 7)
        self.conv8 = SRCNN(n_channels * 8)
        self.conv9 = SRCNN(n_channels * 9)
        self.conv10 = SRCNN(n_channels * 10)

        self.maxPool3d = nn.MaxPool3d((3, 1, 1), stride=(3, 1, 1))

        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out1 = torch.cat([x, out1], dim=1)

        out2 = self.relu(self.conv2(out1))
        out2 = torch.cat([out1, out2], dim=1)

        out3 = self.relu(self.conv3(out2))
        out3 = torch.cat([out2, out3], dim=1)

        out4 = self.relu(self.conv4(out3))
        out4 = torch.cat([out3, out4], dim=1)

        out5 = self.relu(self.conv5(out4))
        out5 = torch.cat([out4, out5], dim=1)

        out6 = self.relu(self.conv6(out5))
        out6 = torch.cat([out5, out6], dim=1)

        out7 = self.relu(self.conv7(out6))
        out7 = torch.cat([out6, out7], dim=1)

        out8 = self.relu(self.conv8(out7))
        out8 = torch.cat([out7, out8], dim=1)

        out9 = self.relu(self.conv9(out8))
        out9 = torch.cat([out8, out9], dim=1)

        out10 = self.relu(self.conv10(out9))

        return out10