"""Prints parameter counts quoted in the paper (21 channels: 1954 total, 514 in the output layer)."""
from model import EEGNet

for c in (20, 21):
    m = EEGNet(c)
    total = sum(p.numel() for p in m.parameters())
    head = sum(p.numel() for p in m.classifier.parameters())
    print(f"{c} channels: total trainable parameters = {total}, output layer = {head}")
