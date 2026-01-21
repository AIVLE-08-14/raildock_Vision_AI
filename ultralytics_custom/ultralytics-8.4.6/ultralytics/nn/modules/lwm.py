# ultralytics/nn/modules/lwm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LWM(nn.Module):
    """
    Lightweight Adaptive Weight Pooling (LWM)
    Paper: YOLO-LAF
    """

    def __init__(self, c):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(c, c // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(c // 4, c, kernel_size=1, bias=False)
        self.act = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.conv1(w)
        w = self.act(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        return x * w
