import os, math, argparse, random, sys
from pathlib import Path
from multiprocessing import freeze_support
from typing import Optional, Callable

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import matplotlib.pyplot as plt
import numpy as np

from .dataset import VectorWatermarkSet
from .encoder import AdvVectorEncoder
from .decoder import AdvVectorDecoder
from .noise_layers import Compose, GaussianNoise, Quantize, DimMask
from configs.config import Config


# ───────── 动态 λ_MSE ─────────
def lambda_mse(epoch: int, total: int) -> float:
    return max(0.5, 2 * (1 - epoch / total))


# ───────── 评估函数 ─────────
@torch.no_grad()
def evaluate(model_tuple, loader, noise, lam, device):
    enc, dec = model_tuple
    enc.eval()
    dec.eval()
    tot_loss = tot_bce = tot_ber = 0.0
    n = 0
    for cover, msg in loader:
        cover = cover.to(device).float()
        msg = msg.to(device).float()
        
        # ⚠️ 关键修复：添加归一化处理（与训练时保持一致）
        norm = torch.norm(cover, p=2, dim=-1, keepdim=True)
        cover = cover / (norm + 1e-8)

        stego = enc(cover, msg)
        stego_noisy = noise(stego)  # 在评估中应用噪声
        logits = dec(stego_noisy)

        probs = torch.sigmoid(logits.float())  # 只 sigmoid 一次
        bce = F.binary_cross_entropy_with_logits(
            logits.float(),  # AMP 自动处理精度
            msg  # 目标 bits (0/1)
        )
        mse = F.mse_loss(torch.tanh(logits) * 0, torch.tanh(logits) * 0)  # dummy
        loss = bce + lam * mse  # mse 此处无用，只保持公式一致
        ber = (probs > .5).float().ne(msg).float().mean()

        tot_loss += loss.item()
        tot_bce += bce.item()
        tot_ber += ber.item()
        n += 1
    return tot_loss / n, tot_bce / n, tot_ber / n


def get_adaptive_model_params(vec_dim: int, msg_len: int):
    base_capacity = vec_dim * msg_len  # 信息容量基准

    # 自适应计算模型深度 (4-16层)
    # 高维向量需要更深的网络来处理复杂特征，支持到8K维度
    # 低维向量用浅网络避免过拟合
    depth = max(4, min(16, int(4 + 12 * (vec_dim / 1024))))

    # 自适应计算隐层倍数 (2-10倍)
    # 高维向量可以用更大的隐层来充分利用信息
    # 低维向量用较小隐层提高参数效率
    base_mul = 2 + 8 * (vec_dim / 2048)  # 线性增长，支持更高维度
    hidden_mul = max(2, min(10, int(base_mul)))

    # 自适应计算扰动强度 (0.005-0.08)
    # 高维向量可以承受更大扰动，但要控制在合理范围内
    delta_scale = max(0.005, min(0.08, 0.01 + 0.07 * (vec_dim / 2048)))

    # 自适应计算dropout率 (0.01-0.4)
    # 超高维向量需要更强的正则化防止过拟合
    dropout = max(0.01, min(0.4, 0.05 + 0.35 * (vec_dim / 2048)))

    return {
        'depth': depth,
        'hidden_mul': hidden_mul,
        'delta_scale': delta_scale,
        'dropout': dropout,
        'capacity_ratio': base_capacity / (vec_dim * vec_dim)  # 用于后续调整
    }


def get_adaptive_training_params(vec_dim: int, base_lr: float):
    """
    根据向量维度自适应计算训练参数

    Args:
        vec_dim: 向量维度
        base_lr: 基础学习率

    Returns:
        dict: 包含训练参数的字典
    """
    # 学习率自适应 (支持到8K维度)
    # 高维向量梯度更复杂，需要更小的学习率
    # 超高维向量需要极小的学习率保证稳定性
    lr_scale = max(0.2, min(2.0, 1.0 * (512 / vec_dim)))  # 扩展范围
    enc_lr = base_lr * lr_scale * 1.2  # 编码器稍高
    dec_lr = base_lr * lr_scale * 0.8  # 解码器稍低

    # 权重衰减自适应 (1e-7 到 5e-4)
    # 超高维向量参数量巨大，需要更强的正则化
    weight_decay = max(1e-7, min(5e-4, 1e-6 * math.sqrt(vec_dim / 64)))

    # 清洁训练比例自适应 (15%-60%)
    # 超高维向量极易过拟合，需要大量清洁训练
    clean_ratio = max(0.15, min(0.6, 0.2 + 0.4 * (vec_dim / 2048)))

    return {
        'enc_lr': enc_lr,
        'dec_lr': dec_lr,
        'weight_decay': weight_decay,
        'clean_ratio': clean_ratio
    }


def get_adaptive_noise_params(vec_dim: int):
    """
    根据向量维度自适应计算噪声参数

    Args:
        vec_dim: 向量维度

    Returns:
        dict: 包含噪声参数的字典
    """
    # 噪声强度自适应
    # 低维向量对噪声更敏感，需要更温和的设置
    noise_scale = math.sqrt(vec_dim / 256)

    # 高斯噪声强度
    gauss_base = 0.015 * noise_scale
    gauss_levels = [gauss_base * 0.5, gauss_base, gauss_base * 1.5]

    # 量化等级
    quant_base = int(8 + 4 * noise_scale)
    quant_levels = [quant_base + 4, quant_base, max(6, quant_base - 2)]

    # 维度遮蔽比例
    mask_base = 0.95 + 0.03 * (1 - noise_scale)
    mask_levels = [min(0.99, mask_base + 0.02), mask_base, max(0.85, mask_base - 0.05)]

    return {
        'gauss_levels': gauss_levels,
        'quant_levels': quant_levels,
        'mask_levels': mask_levels,
        'val_gauss': gauss_base,
        'val_quant': quant_base
    }


# ───────── 从数据库训练的函数 ─────────
def train_from_database(vectors: np.ndarray, vec_dim: int, db_type: str = "unknown", 
                       epochs: Optional[int] = None, learning_rate: Optional[float] = None, 
                       batch_size: Optional[int] = None, val_ratio: Optional[float] = None,
                       progress_callback: Optional[Callable] = None):
    """从数据库数据训练模型
    
    Args:
        vectors: 从数据库获取的向量数据 (N, D)
        vec_dim: 向量维度
        db_type: 数据库类型标识，用于保存路径
        epochs: 训练轮数，默认使用Config.EPOCHS
        learning_rate: 学习率，默认使用Config.LEARNING_RATE
        batch_size: 批处理大小，默认使用Config.BATCH_SIZE
        val_ratio: 验证集比例，默认使用Config.VAL_RATIO
        progress_callback: 进度回调函数
    """
    # 使用提供的参数或默认值
    if epochs is None:
        epochs = Config.EPOCHS
    if learning_rate is None:
        learning_rate = Config.LEARNING_RATE
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if val_ratio is None:
        val_ratio = Config.VAL_RATIO
    
    random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    
    # 根据向量维度创建结果目录
    exp_dir = Config.get_results_dir(vec_dim)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # 使用数据库数据创建数据集
    full_set = VectorWatermarkSet(vectors, Config.MSG_LEN)
    val_len = int(len(full_set) * val_ratio)
    train_len = len(full_set) - val_len
    train_set, val_set = random_split(full_set, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(Config.SEED))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 🚀 自适应参数计算
    model_params = get_adaptive_model_params(vec_dim, Config.MSG_LEN)
    training_params = get_adaptive_training_params(vec_dim, learning_rate)
    noise_params = get_adaptive_noise_params(vec_dim)
    
    print(f"🔧 自适应参数配置 (维度: {vec_dim}):")
    print(f"  模型深度: {model_params['depth']}, 隐层倍数: {model_params['hidden_mul']}")
    print(f"  扰动强度: {model_params['delta_scale']:.4f}, Dropout: {model_params['dropout']:.3f}")
    print(f"  编码器学习率: {training_params['enc_lr']:.2e}, 解码器学习率: {training_params['dec_lr']:.2e}")
    print(f"  清洁训练比例: {training_params['clean_ratio']:.2%}")

    # 模型 - 使用自适应参数
    enc = AdvVectorEncoder(
        vec_dim, 
        Config.MSG_LEN, 
        depth=model_params['depth'],
        hidden_mul=model_params['hidden_mul'],
        delta_scale=model_params['delta_scale']
    ).to(device)
    
    dec = AdvVectorDecoder(
        vec_dim, 
        Config.MSG_LEN,
        depth=model_params['depth'],
        hidden_mul=model_params['hidden_mul'],
        p_drop=model_params['dropout']
    ).to(device)

    # 优化器 - 使用自适应学习率
    enc_opt = torch.optim.Adam(
        enc.parameters(), 
        lr=training_params['enc_lr'], 
        betas=(0.9, 0.999),
        weight_decay=training_params['weight_decay']
    )
    dec_opt = torch.optim.Adam(
        dec.parameters(), 
        lr=training_params['dec_lr'], 
        betas=(0.9, 0.999),
        weight_decay=training_params['weight_decay']
    )

    total_steps = len(train_loader) * epochs
    warm_steps = int(0.05 * total_steps)
    lr_lambda = lambda s: (s / warm_steps if s < warm_steps else
                           0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm_steps) / (total_steps - warm_steps))))
    enc_sched = torch.optim.lr_scheduler.LambdaLR(enc_opt, lr_lambda)
    dec_sched = torch.optim.lr_scheduler.LambdaLR(dec_opt, lr_lambda)
    scaler = GradScaler(device, enabled=(device == "cuda"))

    # ---------- 自适应噪声配置 ----------
    val_noise = Compose((
        GaussianNoise(noise_params['val_gauss']), 
        Quantize(noise_params['val_quant'])
    )).to(device)

    noise_pool = []
    # 添加不同强度的高斯噪声
    for gauss_level in noise_params['gauss_levels']:
        noise_pool.append(GaussianNoise(gauss_level))
    
    # 添加不同精度的量化噪声
    for quant_level in noise_params['quant_levels']:
        noise_pool.append(Quantize(quant_level))
    
    # 添加不同比例的维度遮蔽
    for mask_level in noise_params['mask_levels']:
        noise_pool.append(DimMask(mask_level))
    
    for n in noise_pool:
        n.to(device)

    print(f"🔧 噪声配置: {len(noise_pool)}种噪声类型")
    print(f"  验证噪声: Gauss({noise_params['val_gauss']:.3f}) + Quant({noise_params['val_quant']})")

    # 指标列表
    tr_loss, tr_bce, tr_ber = [], [], []
    va_loss, va_bce, va_ber = [], [], []

    best_val_ber = 1e9
    stall = 0

    print(f"开始训练 {vec_dim}维向量模型，数据来源：{db_type}")
    print(f"训练数据：{len(train_set)} 样本，验证数据：{len(val_set)} 样本")
    print(f"训练参数：epochs={epochs}, lr={learning_rate}, batch_size={batch_size}, val_ratio={val_ratio}")

    for ep in range(1, epochs + 1):
        # 动态噪声策略选择
        max_noises_to_compose = min(1 + (ep - 1) // (epochs // 4), 3)

        # 🔧 自适应delta_scale动态调整
        base_delta = model_params['delta_scale']
        growth_factor = 1 + 0.4 * ep / epochs
        enc.delta_scale = min(base_delta * 1.5, base_delta * growth_factor)
            
        lam = lambda_mse(ep, epochs)

        enc.train()
        dec.train()
        ep_l = ep_b = ep_r = 0.0
        n_batch = 0
        
        for cover, msg in train_loader:
            cover = cover.to(device).float()
            msg = msg.to(device).float()
            norm = torch.norm(cover, p=2, dim=-1, keepdim=True)
            cover = cover / (norm + 1e-8)

            # 🔧 自适应噪声应用策略
            if random.random() < training_params['clean_ratio']:
                # 清洁训练，比例根据维度自适应
                noise = nn.Identity().to(device)
            else:
                # 噪声训练，组合数量有限制
                num_noises = random.randint(1, min(max_noises_to_compose, len(noise_pool) // 3))
                noises_to_apply = random.sample(noise_pool, k=num_noises)
                noise = Compose(tuple(noises_to_apply))

            with autocast(device, enabled=(device == "cuda")):
                stego = enc(cover, msg)
                stego_noisy = noise(stego)
                logits = dec(stego_noisy)
                probs = torch.sigmoid(logits.float())
                bce = F.binary_cross_entropy_with_logits(
                    logits.float(),
                    msg
                )
                mse = F.mse_loss(stego, cover)
                loss = bce + lam * mse
                
            scaler.scale(loss).backward()
            scaler.unscale_(enc_opt)
            scaler.unscale_(dec_opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            scaler.step(enc_opt)
            scaler.step(dec_opt)
            scaler.update()
            enc_opt.zero_grad(set_to_none=True)
            dec_opt.zero_grad(set_to_none=True)
            enc_sched.step()
            dec_sched.step()

            ber = (probs > .5).float().ne(msg).float().mean()

            ep_l += loss.item()
            ep_b += bce.item()
            ep_r += ber.item()
            n_batch += 1

        tr_loss.append(ep_l / n_batch)
        tr_bce.append(ep_b / n_batch)
        tr_ber.append(ep_r / n_batch)

        # 验证
        v_loss, v_bce, v_ber = evaluate((enc, dec), val_loader, val_noise, lam, device)
        va_loss.append(v_loss)
        va_bce.append(v_bce)
        va_ber.append(v_ber)

        print(f"E{ep:03d} | Train loss {tr_loss[-1]:.4f}  BCE {tr_bce[-1]:.4f}  BER {tr_ber[-1]:.3%} "
              f"|| Val loss {v_loss:.4f}  BCE {v_bce:.4f}  BER {v_ber:.3%}")

        # 调用进度回调
        if progress_callback:
            progress_callback(ep, epochs, {
                'train_loss': tr_loss[-1],
                'train_bce': tr_bce[-1], 
                'train_ber': tr_ber[-1],
                'val_loss': v_loss,
                'val_bce': v_bce,
                'val_ber': v_ber
            })

        # Early-stop on val BER
        if v_ber < best_val_ber - 0.0005:
            best_val_ber = v_ber
            stall = 0
            model_path = Config.get_model_path(vec_dim)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()}, model_path)
            print(f"保存最佳模型到: {model_path}")

    print(f"训练完成！最佳验证BER: {best_val_ber:.3%}")
    
    # 🔧 自适应性能评估标准
    # 根据向量维度和模型容量调整评估标准
    capacity_ratio = model_params['capacity_ratio']
    
    # 动态调整性能阈值
    excellent_threshold = max(0.005, min(0.02, 0.01 * (1 + capacity_ratio)))
    good_threshold = max(0.02, min(0.08, 0.05 * (1 + capacity_ratio)))
    
    performance_level = "excellent" if best_val_ber < excellent_threshold else \
                       "good" if best_val_ber < good_threshold else "poor"
    
    suggestions = []
    
    if performance_level == "poor":
        # 通用建议
        if best_val_ber > good_threshold * 2:
            suggestions.append(f"验证错误率({best_val_ber:.1%})过高，建议增加训练轮数或调整学习率")
        if len(train_set) < vec_dim * 50:  # 基于维度的最小样本数要求
            suggestions.append(f"训练数据较少({len(train_set)}样本)，建议增加到至少{vec_dim * 100}样本")
        
        # 基于模型容量的建议
        if model_params['hidden_mul'] < 6:
            suggestions.append("模型容量可能不足，考虑增加hidden_mul参数")
        if training_params['clean_ratio'] > 0.3:
            suggestions.append("清洁训练比例较高，可以尝试增加噪声训练")
        
        # 通用优化建议
        suggestions.extend([
            f"针对{vec_dim}维向量的优化建议:",
            "1. 确保训练数据质量高且分布均匀",
            "2. 可尝试增加批处理大小以提高训练稳定性", 
            "3. 检查数据预处理是否正确（L2归一化等）",
            "4. 考虑调整验证集比例为0.1-0.2之间",
            f"5. 当前模型使用{model_params['depth']}层深度，可尝试±1层调整"
        ])
        
    elif performance_level == "good":
        suggestions.append(f"{vec_dim}维向量训练效果良好！如需进一步优化，可适当增加训练轮数")
    
    # 性能评估反馈
    print(f"\n📊 性能评估 (维度: {vec_dim}):")
    print(f"  性能等级: {performance_level.upper()}")
    print(f"  评估阈值: 优秀<{excellent_threshold:.1%}, 良好<{good_threshold:.1%}")
    print(f"  模型配置: {model_params['depth']}层 × {model_params['hidden_mul']}倍隐层")
    
    if performance_level == "excellent":
        print(f"🎉 {vec_dim}维向量达到优秀水平！")
    elif performance_level == "good":
        print(f"✅ {vec_dim}维向量训练成功！")
    else:
        print(f"⚠️  {vec_dim}维向量训练需要优化，建议参考以下建议")
    
    return {
        "success": True,
        "best_ber": best_val_ber,
        "model_path": Config.get_model_path(vec_dim),
        "epochs": epochs,
        "performance_level": performance_level,
        "suggestions": suggestions,
        "adaptive_config": {  # 记录使用的自适应配置
            "model_params": model_params,
            "training_params": training_params,
            "noise_params": noise_params
        },
        "evaluation_thresholds": {
            "excellent": excellent_threshold,
            "good": good_threshold
        },
        "final_metrics": {
            "train_loss": tr_loss[-1] if tr_loss else 0,
            "train_ber": tr_ber[-1] if tr_ber else 1,
            "val_loss": va_loss[-1] if va_loss else 0,
            "val_ber": best_val_ber
        }
    }


# ────────────────────────
if __name__ == "__main__":
    freeze_support()
    # Example usage (replace with actual data and parameters)
    # vectors = np.load("path/to/your/vectors.npy") # Replace with your data path
    # vec_dim = 384 # Replace with your vector dimension
    # db_type = "example_db"
    # result = train_from_database(vectors, vec_dim, db_type=db_type, epochs=100, learning_rate=0.001, batch_size=64, val_ratio=0.2)
    # print(result)
