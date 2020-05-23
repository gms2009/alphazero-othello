import torch
import torch.nn as nn

from typing import Tuple


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class ResidualModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._convs = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self._relu = nn.ReLU()

    def forward(self, x):
        z = self._convs(x)
        x = x + z
        x = self._relu(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(17, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self._residuals = nn.Sequential(
            *[ResidualModule() for _ in range(5)]
        )
        self._policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(128, 65),
            nn.Softmax(dim=1)
        )
        self._value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self._conv(images)
        images = self._residuals(images)
        p = self._policy_head(images)
        v = self._value_head(images)
        return p, v

    def inference(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image.unsqueeze_(0)
        p, v = self.forward(image)
        p.squeeze_(0)
        v.squeeze_(0)
        return p, v
