import argparse, os, random, json, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from vector_dataset   import VectorWatermarkSet
from model.vector_encoder   import AdvVectorEncoder     # :contentReference[oaicite:0]{index=0}
from model.vector_decoder   import AdvVectorDecoder     # :contentReference[oaicite:1]{index=1}
from noise_layers.vnoise_layers    import Compose              # :contentReference[oaicite:2]{index=2}
from noise_argparser  import NoiseArgParser
from multiprocessing import freeze_support

# ─── 1. 解析参数 ──────────────────────────
def main():
    # ─── 1. 解析参数 ──────────────────────────
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="data/cover_vectors.npy")
    p.add_argument("--msg_len", type=int, default=96)
    p.add_argument("--vec_dim", type=int, default=384)
    p.add_argument("--batch",   type=int, default=512)
    p.add_argument("--epochs",  type=int, default=60)
    p.add_argument(
        "--noise",
        nargs=1,
        action=NoiseArgParser,
        default=["gauss(0.02)+quantize(8)+mask(0.9)"],
        help="'gauss(0.02)+quantize(8)+mask(0.9)'"
    )
    p.add_argument("--exp_dir", default="experiments/vector_run1")
    args = p.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)

    # ─── 2. 数据加载 ──────────────────────────
    dataset = VectorWatermarkSet(args.data, args.msg_len)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,        # Windows 下多进程，需要入口保护
        pin_memory=True
    )

    # ─── 3. 模型与优化器 ────────────────────────
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    enc      = AdvVectorEncoder(args.vec_dim, args.msg_len).to(device)
    dec      = AdvVectorDecoder(args.vec_dim, args.msg_len).to(device)

    # 噪声层
    noise_net = Compose(tuple(args.noise)).to(device)

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=3e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-6
    )
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    # ─── 4. 训练循环 ──────────────────────────
    for epoch in range(1, args.epochs + 1):
        enc.train(); dec.train()
        tot_msg = tot_dist = 0.0

        for cover, msg in loader:
            cover = cover.to(device)
            msg   = msg.to(device)

            stego  = enc(cover, msg)
            noised = noise_net(stego)
            pred   = dec(noised)

            loss_msg  = bce(pred, msg)
            loss_dist = mse(stego, cover) * 2.0
            loss      = loss_msg + loss_dist

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_msg  += loss_msg.item()  * cover.size(0)
            tot_dist += loss_dist.item() * cover.size(0)

        N = len(dataset)
        print(f"[Epoch {epoch:02d}/{args.epochs}] "
              f"BCE={tot_msg/N:.4f}  MSE*2={tot_dist/N:.6f}")

        torch.save({
            "epoch": epoch,
            "enc": enc.state_dict(),
            "dec": dec.state_dict(),
            "opt": opt.state_dict()
        }, f"{args.exp_dir}/epoch{epoch:02d}.pt")


if __name__ == "__main__":
    freeze_support()
    main()