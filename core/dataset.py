# vector_dataset.py
import torch, numpy as np
from torch.utils.data import Dataset

class VectorWatermarkSet(Dataset):
    """
    cover_vectors.npy :  shape = (N , 384) , dtype = float32 / float16
    """
    def __init__(self, npy_path: str, msg_len: int, mmap: bool = True):
        self.vecs = np.load(npy_path, mmap_mode="r" if mmap else None)
        assert self.vecs.ndim == 2, "expect (N,D)"
        self.msg_len = msg_len

    def __len__(self):                 # → N
        return self.vecs.shape[0]

    def __getitem__(self, idx):
        cover = torch.from_numpy(self.vecs[idx]).float()

        # —— 1) 随机选一个块索引 k ∈ [0,16) ——
        k = np.random.randint(0, 16)
        idx_bits = [(k >> i) & 1 for i in reversed(range(4))]  # MSB first

        # —— 2) 计算 4 bit CRC-4 校验（多项式 0x3） ——
        reg = 0
        for bit in idx_bits:
            reg ^= (bit << 3)
            for _ in range(4):
                if reg & 0x8:
                    reg = ((reg << 1) & 0xF) ^ 0x3
                else:
                    reg = (reg << 1) & 0xF
        crc_bits = [(reg >> i) & 1 for i in reversed(range(4))]

        # —— 3) 随机生成 16 bit payload ——
        payload = np.random.randint(0, 2, size=(16,), dtype=np.uint8).tolist()

        # —— 4) 拼成 24 bit 消息 ——
        msg_bits = idx_bits + crc_bits + payload  # 列表长度正好 24
        msg = torch.tensor(msg_bits, dtype=torch.float32)

        return cover, msg
