import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------  模块：FiLM 残差块 ----------
class FiLMResidualBlock(nn.Module):
    """
    残差 MLP 块 + Feature-wise Linear Modulation (FiLM)
    γ, β 由 message MLP 产生，用于调制 cover 分支激活
    """

    def __init__(self, dim: int, msg_dim: int, hidden: int):
        super().__init__()
        # 载体向量
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln = nn.LayerNorm(dim)
        # message → gamma, beta
        self.msg_fc = nn.Sequential(
            nn.Linear(msg_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim * 2)  # γ || β
        )

    def forward(self, x, m):
        """
        x : (N, D)  cover / 先前隐向量
        m : (N, L)  message
        """
        gamma_beta = self.msg_fc(m)  # (N, 2D)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)  # (N, D)

        # FiLM 调制
        h = (1 + gamma) * h + beta
        return x + h  # 残差


# ----------  Encoder 主体 ----------
class AdvVectorEncoder(nn.Module):
    """
    - 深层残差 + LayerNorm + GELU
    - FiLM 用 message 条件化 cover 分支
    """

    def __init__(
            self,
            vec_dim: int,  # D
            msg_len: int,  # L
            depth: int = 6,
            hidden_mul: int = 4,
            delta_scale: float = 0.02
    ):
        super().__init__()
        self.delta_scale = delta_scale
        hidden = vec_dim * hidden_mul

        self.blocks = nn.ModuleList([
            FiLMResidualBlock(vec_dim, msg_len, hidden)
            for _ in range(depth)
        ])

        self.out_ln = nn.LayerNorm(vec_dim)
        self.out_fc = nn.Linear(vec_dim, vec_dim)

    def forward(self, cover_vec: torch.Tensor, message: torch.Tensor):
        h = cover_vec
        for blk in self.blocks:
            h = blk(h, message)

        delta = torch.tanh(self.out_fc(self.out_ln(h))) * self.delta_scale
        return cover_vec + delta  # stego_vec
