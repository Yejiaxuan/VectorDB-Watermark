import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, p_drop: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln  = nn.LayerNorm(dim)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


class AdvVectorDecoder(nn.Module):
    """
    - 多层残差 MLP + LayerNorm
    - 最终 Sigmoid 输出比特概率
    """
    def __init__(
        self,
        vec_dim: int,
        msg_len: int,
        depth: int = 6,
        hidden_mul: int = 4,
        p_drop: float = 0.1
    ):
        super().__init__()
        hidden = vec_dim * hidden_mul

        self.blocks = nn.ModuleList([
            ResidualMLPBlock(vec_dim, hidden, p_drop)
            for _ in range(depth)
        ])

        self.out_ln = nn.LayerNorm(vec_dim)
        self.out_fc = nn.Linear(vec_dim, msg_len)

    def forward(self, stego_vec: torch.Tensor):
        h = stego_vec
        for blk in self.blocks:
            h = blk(h)

        logits = self.out_fc(self.out_ln(h))
        return torch.sigmoid(logits)           # (N, L)
