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

    def __getitem__(self, idx):        # → (cover_vec , message_bits)
        cover = torch.from_numpy(self.vecs[idx]).float()   # (384,)
        msg   = torch.randint(0, 2, (self.msg_len,), dtype=torch.float32)
        return cover, msg
