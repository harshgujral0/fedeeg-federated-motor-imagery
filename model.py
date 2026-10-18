"""EEGNet-8,2 with named modules so that FL methods can pick keys by prefix.

Key groups used by the federated methods:
  head      -> keys starting with "classifier"
  batchnorm -> keys starting with "bn"
Max-norm constraints from the original EEGNet are omitted (as in many ports).
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_chans, n_times=512, n_classes=2, F1=8, D=2, F2=16,
                 kern=64, p=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, F1, (1, kern), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(p)
        self.sep_dw = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same",
                                groups=F1 * D, bias=False)
        self.sep_pw = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(p)
        self.act = nn.ELU()
        self.classifier = nn.Linear(F2 * (n_times // 4 // 8), n_classes)

    def features(self, x):                       # x: (B, C, T)
        x = x.unsqueeze(1)
        x = self.bn1(self.temporal(x))
        x = self.act(self.bn2(self.spatial(x)))
        x = self.drop1(self.pool1(x))
        x = self.act(self.bn3(self.sep_pw(self.sep_dw(x))))
        x = self.drop2(self.pool2(x))
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self.features(x))


def is_head(key):
    return key.startswith("classifier")


def is_bn(key):
    return key.startswith("bn")
