#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vector.py  –  向量隐写训练 + 训练/验证曲线
"""

import os, math, argparse, random
from pathlib import Path
from multiprocessing import freeze_support

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch import amp
import matplotlib.pyplot as plt

from vector_dataset  import VectorWatermarkSet
from src.vector_encoder  import AdvVectorEncoder
from src.vector_decoder  import AdvVectorDecoder
from src.vnoise_layers   import Compose, GaussianNoise, Quantize, DimMask


# ───────── CLI ─────────
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data",        default="HiDDeN/nq_qa_combined_384d.npy", help="向量 npy/pt 文件")
    p.add_argument("--msg_len",     type=int, default=96)
    p.add_argument("--vec_dim",     type=int, default=384)
    p.add_argument("--batch",       type=int, default=8192)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--exp_dir",     default="results/vector_val")
    p.add_argument("--val_ratio",   type=float, default=0.15, help="验证集比例")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ───────── 动态 λ_MSE ─────────
def lambda_mse(epoch: int, total: int) -> float:
    return max(0.5, 2 * (1 - epoch / total))


# ───────── 评估函数 ─────────
@torch.no_grad()
def evaluate(model_tuple, loader, noise, lam, device):
    enc, dec = model_tuple
    enc.eval(); dec.eval()
    tot_loss = tot_bce = tot_ber = 0.0; n = 0
    for cover, msg in loader:
        cover = cover.to(device).float()
        msg   = msg.to(device).float()
        logits = dec(enc(cover, msg))
        probs = torch.sigmoid(logits.float())  # 只 sigmoid 一次
        bce = F.binary_cross_entropy_with_logits(
            logits.float(),  # AMP 自动处理精度
            msg  # 目标 bits (0/1)
        )
        mse = F.mse_loss(torch.tanh(logits)*0, torch.tanh(logits)*0)  # dummy
        loss = bce + lam * mse     # mse 此处无用，只保持公式一致
        ber = (probs > .5).float().ne(msg).float().mean()


        tot_loss += loss.item(); tot_bce += bce.item(); tot_ber += ber.item()
        n += 1
    return tot_loss / n, tot_bce / n, tot_ber / n


# ───────── 主函数 ─────────
def main():
    args = get_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)

    # 数据加载与划分
    full_set = VectorWatermarkSet(args.data, args.msg_len)
    val_len  = int(len(full_set) * args.val_ratio)
    train_len = len(full_set) - val_len
    train_set, val_set = random_split(full_set, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    # 模型
    enc = AdvVectorEncoder(args.vec_dim, args.msg_len, delta_scale=0.02).to(device)
    dec = AdvVectorDecoder(args.vec_dim, args.msg_len).to(device)

    enc_opt = torch.optim.Adam(enc.parameters(), lr=args.lr * 2, betas=(0.9, 0.999))
    dec_opt = torch.optim.Adam(dec.parameters(), lr=args.lr,       betas=(0.9, 0.999))

    total_steps = len(train_loader) * args.epochs
    warm_steps  = int(0.05 * total_steps)
    lr_lambda   = lambda s: (s/warm_steps if s<warm_steps else
                             0.1 + 0.9*0.5*(1+math.cos(math.pi*(s-warm_steps)/(total_steps-warm_steps))))
    enc_sched = torch.optim.lr_scheduler.LambdaLR(enc_opt, lr_lambda)
    dec_sched = torch.optim.lr_scheduler.LambdaLR(dec_opt, lr_lambda)
    scaler    = amp.GradScaler(enabled=(device=="cuda"))

    # 指标列表
    tr_loss, tr_bce, tr_ber = [], [], []
    va_loss, va_bce, va_ber = [], [], []

    best_val_ber = 1e9; stall = 0

    for ep in range(1, args.epochs + 1):

        # ---------- 动态噪声 ----------
        if ep <= 20:
            noise = Compose((GaussianNoise(0.02),)).to(device)
        elif ep <= 40:
            noise = Compose((GaussianNoise(0.02), Quantize(12))).to(device)
        elif ep <= 60:
            noise = Compose((GaussianNoise(0.02), Quantize(10))).to(device)
        elif ep <= 80:
            noise = Compose((GaussianNoise(0.02), Quantize(8))).to(device)
        else:
            noise = Compose((GaussianNoise(0.02), Quantize(8), DimMask(0.9))).to(device)

        enc.delta_scale = min(0.03, 0.02*(1+0.5*ep/args.epochs))
        lam = lambda_mse(ep, args.epochs)

        enc.train(); dec.train()
        ep_l = ep_b = ep_r = 0.0; n_batch = 0
        for cover, msg in train_loader:
            cover = cover.to(device).float()
            msg   = msg.to(device).float()
            norm = torch.norm(cover, p=2, dim=-1, keepdim=True)
            cover = cover / (norm + 1e-8)

            with amp.autocast(enabled=(device=="cuda"),device_type="cuda"):
                stego  = enc(cover, msg)
                logits = dec(stego)
                probs = torch.sigmoid(logits.float())  # 只 sigmoid 一次
                bce = F.binary_cross_entropy_with_logits(
                    logits.float(),  # AMP 自动处理精度
                    msg  # 目标 bits (0/1)
                )
                mse = F.mse_loss(stego, cover)
                loss = bce + lam * mse
            scaler.scale(loss).backward()
            scaler.unscale_(enc_opt); scaler.unscale_(dec_opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            scaler.step(enc_opt); scaler.step(dec_opt); scaler.update()
            enc_opt.zero_grad(set_to_none=True); dec_opt.zero_grad(set_to_none=True)
            enc_sched.step(); dec_sched.step()

            ber = (probs > .5).float().ne(msg).float().mean()

            ep_l += loss.item(); ep_b += bce.item(); ep_r += ber.item(); n_batch += 1

        tr_loss.append(ep_l/n_batch); tr_bce.append(ep_b/n_batch); tr_ber.append(ep_r/n_batch)

        # ---------- 验证 ----------
        v_loss, v_bce, v_ber = evaluate((enc, dec), val_loader, noise, lam, device)
        va_loss.append(v_loss); va_bce.append(v_bce); va_ber.append(v_ber)

        print(f"E{ep:03d} | Train loss {tr_loss[-1]:.4f}  BCE {tr_bce[-1]:.4f}  BER {tr_ber[-1]:.3%} "
              f"|| Val loss {v_loss:.4f}  BCE {v_bce:.4f}  BER {v_ber:.3%}")

        # Early-stop on val BER
        if v_ber < best_val_ber - 0.0005:
            best_val_ber = v_ber; stall = 0
            torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()},
                       Path(args.exp_dir)/"best.pt")
        #else:
            #stall += 1
            #if stall >= 5:
                #print(f"Early stopped at epoch {ep}")
                #break

    # ---------- 绘图 ----------
    epochs = range(1, len(tr_loss) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex="all")

    # 1. Total Loss
    axes[0].plot(epochs, tr_loss, label="Train")
    axes[0].plot(epochs, va_loss, label="Val")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss (Train vs. Val)")
    axes[0].grid(True)
    axes[0].legend()

    # 2. BCE
    axes[1].plot(epochs, tr_bce, label="Train")
    axes[1].plot(epochs, va_bce, label="Val")
    axes[1].set_ylabel("BCE")
    axes[1].set_title("BCE (Train vs. Val)")
    axes[1].grid(True)
    axes[1].legend()

    # 3. BER  (百分比显示更直观)
    axes[2].plot(epochs, [x * 100 for x in tr_ber], label="Train")
    axes[2].plot(epochs, [x * 100 for x in va_ber], label="Val")
    axes[2].set_ylabel("BER (%)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("BER (Train vs. Val)")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()  # 若在无 GUI 环境，可注释掉



# ────────────────────────
if __name__ == "__main__":
    freeze_support()
    main()
