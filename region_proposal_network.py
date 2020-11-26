import torch
import torch.nn as nn

class RNP(nn.Module):
    def __init__(self, backbone,
                 aspect_ratios=[(1, 1), (1, 2), (2, 1)],
                 scales=3):

    def forward(self, x):