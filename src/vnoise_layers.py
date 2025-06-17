# ──────────────────────────────────────────────────────────────
# file: vector_model/vnoise_layers.py
# ──────────────────────────────────────────────────────────────
"""
可微向量失真算子 (vector-domain noise layers)

每个类都实现:
    forward(x):  x.shape == (B, d)  →  same shape
并且在 __all__ 中暴露，供噪声字符串解析器自动注册。
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Tuple

#    加性高斯噪声  x ← x + N(0, σ²)
class GaussianNoise(nn.Module):
    def __init__(self, sigma: float = 0.01):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            torch.set_grad_enabled(True)
        noise = torch.randn_like(x) * self.sigma
        return x + noise



#    量化  x ← round(x·2^k)/2^k

class Quantize(nn.Module):
    def __init__(self, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.scale = 2 ** n_bits - 1

    def forward(self, x):
        # 将 [-1,1] → [0,1]，量化，再还原
        x_scaled = (x + 1) / 2
        x_q = torch.round(x_scaled * self.scale) / self.scale
        x_q = x_q * 2 - 1
        # Straight-Through Estimator
        return x + (x_q - x).detach()


# 3   随机维度丢弃  x_i ← 0  with prob (1 − keep_prob)

class DimMask(nn.Module):
    def __init__(self, keep_prob: float = 0.9):
        super().__init__()
        self.keep = keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.keep >= 1.0:
            return x
        mask = torch.rand_like(x) < self.keep
        return x * mask.float()

#    随机正交旋转 / 投影  x ← Rx
#    R ∈ O(d)  每个 batch 重新采样
class RandProj(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _random_orthogonal(d: int, device) -> torch.Tensor:
        # 生成随机高斯 → QR 分解 → 取正交矩阵
        a = torch.randn(d, d, device=device)
        q, r = torch.linalg.qr(a, mode="reduced")
        # 确保 det(R)=1  (不影响距离，但保持旋转而非旋转+反射)
        diag = torch.diag(r)
        q = q * diag.sign()
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, d = x.shape
        R = self._random_orthogonal(d, x.device)
        return torch.matmul(x, R)             # (B, d)·(d, d)


class Compose(nn.Module):
    def __init__(self, layers: Tuple[nn.Module, ...]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


__all__ = [
    "GaussianNoise",
    "Quantize",
    "DimMask",
    "RandProj",
    "Compose",
]
