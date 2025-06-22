"""
向量水印性能测试脚本 - 评估嵌入质量、提取准确率、抗噪声能力和时间效率
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

# 导入自定义模块
from algorithms.deep_learning.watermark import VectorWatermark
from algorithms.deep_learning.noise_layers import (
    Compose, GaussianNoise, Quantize, DimMask
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="向量水印性能测试")
    parser.add_argument("--model_path", type=str, default="algorithms/deep_learning/results/vector_val/best.pt", 
                        help="预训练模型路径")
    parser.add_argument("--data_path", type=str, default="algorithms/deep_learning/dataset/qa/nq_qa_combined_384d.npy", 
                        help="测试数据路径")
    parser.add_argument("--vec_dim", type=int, default=384, help="向量维度")
    parser.add_argument("--msg_len", type=int, default=24, help="消息长度")
    parser.add_argument("--batch_size", type=int, default=128, help="批处理大小")
    parser.add_argument("--test_samples", type=int, default=1000, help="测试样本数量")
    parser.add_argument("--bit_error_threshold", type=int, default=1, 
                        help="容错比特错误阈值")
    parser.add_argument("--device", type=str, default=None, help="计算设备(cpu/cuda)")
    parser.add_argument("--output_dir", type=str, default="test_results", help="结果输出目录")
    return parser.parse_args()


def load_test_data(data_path: str, n_samples: int) -> np.ndarray:
    """加载测试数据"""
    print(f"正在加载测试数据: {data_path}")
    data = np.load(data_path)
    
    # 如果需要的样本数小于数据集大小，随机选择样本
    if n_samples < len(data):
        indices = np.random.choice(len(data), n_samples, replace=False)
        data = data[indices]
    
    return data


def evaluate_cosine_similarity(
    model: VectorWatermark, 
    test_data: np.ndarray, 
    batch_size: int
) -> Dict:
    """评估嵌入前后的余弦相似度"""
    print("\n评估嵌入质量（余弦相似度）...")
    
    device = model.device
    n_samples = len(test_data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    similarities = []
    
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_data = test_data[start_idx:end_idx]
        
        # 转换为tensor
        cover_vec = torch.tensor(batch_data, dtype=torch.float32, device=device)
        
        # 生成随机消息并嵌入水印
        stego_vec, messages = model.encode(cover_vec, random_msg=True)
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(cover_vec, stego_vec)
        similarities.extend(cos_sim.cpu().numpy())
    
    # 计算统计结果
    similarities = np.array(similarities)
    results = {
        "mean_similarity": float(np.mean(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "std_similarity": float(np.std(similarities)),
    }
    
    return results


def evaluate_accuracy(
    model: VectorWatermark, 
    test_data: np.ndarray, 
    batch_size: int,
    bit_error_threshold: int
) -> Dict:
    """评估水印提取准确率，使用指定的比特错误阈值"""
    print("\n评估水印提取准确率...")
    
    device = model.device
    n_samples = len(test_data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # 存储每个样本的比特错误数
    bit_errors = []
    
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_data = test_data[start_idx:end_idx]
        
        # 转换为tensor
        cover_vec = torch.tensor(batch_data, dtype=torch.float32, device=device)
        
        # 生成随机消息并嵌入水印
        stego_vec, original_msg = model.encode(cover_vec, random_msg=True)
        
        # 从隐写向量中提取消息
        decoded_msg = model.decode(stego_vec)
        
        # 计算每个样本的错误比特数
        errors = (decoded_msg != original_msg).int().sum(dim=1)
        bit_errors.extend(errors.cpu().numpy())
    
    # 计算统计结果
    bit_errors = np.array(bit_errors)
    results = {}
    
    # 总比特错误率
    ber = np.mean(bit_errors) / model.msg_len
    results["bit_error_rate"] = float(ber)
    
    # 指定阈值下的准确率
    accuracy = np.mean(bit_errors <= bit_error_threshold)
    results["accuracy"] = float(accuracy)
    
    return results


def evaluate_crc_verification(
    model: VectorWatermark,
    test_data: np.ndarray,
    batch_size: int
) -> Dict:
    """评估CRC校验成功率"""
    print("\n评估CRC校验成功率...")
    
    device = model.device
    n_samples = len(test_data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # 存储CRC验证结果
    crc_verification_results = []
    
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_data = test_data[start_idx:end_idx]
        
        # 转换为tensor
        cover_vec = torch.tensor(batch_data, dtype=torch.float32, device=device)
        
        # 生成随机消息并嵌入水印
        stego_vec, original_msg = model.encode(cover_vec, random_msg=True)
        
        # 从隐写向量中提取消息
        decoded_msg = model.decode(stego_vec)
        
        # 验证CRC校验和
        verification_results = model.verify_message(decoded_msg)
        if isinstance(verification_results, bool):
            crc_verification_results.append(verification_results)
        else:
            crc_verification_results.extend(verification_results)
    
    # 计算统计结果
    crc_verification_rate = np.mean(crc_verification_results)
    
    results = {
        "crc_verification_rate": float(crc_verification_rate)
    }
    
    return results


def evaluate_noise_robustness(
    model: VectorWatermark, 
    test_data: np.ndarray, 
    batch_size: int,
    bit_error_threshold: int
) -> Dict:
    """评估在不同噪声条件下的水印提取鲁棒性"""
    print("\n评估噪声鲁棒性...")
    
    device = model.device
    n_samples = len(test_data)
    
    # 定义噪声列表，参考trainer.py中的噪声池
    noise_layers = {
        "无噪声": nn.Identity().to(device),
        "高斯噪声(0.01)": GaussianNoise(0.01).to(device),
        "高斯噪声(0.02)": GaussianNoise(0.02).to(device),
        "量化(12)": Quantize(12).to(device),
        "量化(10)": Quantize(10).to(device),
        "量化(8)": Quantize(8).to(device),
        "维度遮蔽(0.95)": DimMask(0.95).to(device),
        "维度遮蔽(0.90)": DimMask(0.9).to(device),
        "组合噪声1": Compose([GaussianNoise(0.01), Quantize(10)]).to(device),
        "组合噪声2": Compose([GaussianNoise(0.02), DimMask(0.95)]).to(device),
        "组合噪声3": Compose([Quantize(8), DimMask(0.9), GaussianNoise(0.01)]).to(device)
    }
    
    results = {}
    
    for noise_name, noise_layer in noise_layers.items():
        print(f"  测试噪声: {noise_name}")
        bit_errors_with_noise = []
        crc_verification_results = []
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        for i in tqdm(range(n_batches), leave=False):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_data = test_data[start_idx:end_idx]
            
            # 转换为tensor
            cover_vec = torch.tensor(batch_data, dtype=torch.float32, device=device)
            
            # 生成随机消息并嵌入水印
            stego_vec, original_msg = model.encode(cover_vec, random_msg=True)
            
            # 应用噪声
            noisy_stego = noise_layer(stego_vec)
            
            # 从带噪声的隐写向量中提取消息
            decoded_msg = model.decode(noisy_stego)
            
            # 计算每个样本的错误比特数
            errors = (decoded_msg != original_msg).int().sum(dim=1)
            bit_errors_with_noise.extend(errors.cpu().numpy())
            
            # 验证CRC校验和
            verification_results = model.verify_message(decoded_msg)
            if isinstance(verification_results, bool):
                crc_verification_results.append(verification_results)
            else:
                crc_verification_results.extend(verification_results)
        
        # 计算噪声条件下的统计数据
        bit_errors_with_noise = np.array(bit_errors_with_noise)
        
        # 比特错误率
        ber = np.mean(bit_errors_with_noise) / model.msg_len
        results[f"{noise_name}_bit_error_rate"] = float(ber)
        
        # 指定阈值下的准确率
        accuracy = np.mean(bit_errors_with_noise <= bit_error_threshold)
        results[f"{noise_name}_accuracy"] = float(accuracy)
        
        # CRC校验成功率
        crc_rate = np.mean(crc_verification_results)
        results[f"{noise_name}_crc_verification_rate"] = float(crc_rate)
    
    return results


def evaluate_performance(
    model: VectorWatermark, 
    test_data: np.ndarray, 
    batch_size: int
) -> Dict:
    """评估嵌入和提取的时间性能"""
    print("\n评估时间性能...")
    
    device = model.device
    n_samples = min(len(test_data), 500)  # 限制样本数，避免评估时间过长
    test_data = test_data[:n_samples]
    
    # 单个样本处理时间
    single_embed_times = []
    single_extract_times = []
    
    print("  测试单样本处理时间...")
    for i in tqdm(range(n_samples)):
        cover_vec = torch.tensor(test_data[i:i+1], dtype=torch.float32, device=device)
        
        # 测量嵌入时间
        start_time = time.time()
        stego_vec, message = model.encode(cover_vec, random_msg=True)
        single_embed_times.append(time.time() - start_time)
        
        # 测量提取时间
        start_time = time.time()
        _ = model.decode(stego_vec)
        single_extract_times.append(time.time() - start_time)
    
    # 批处理时间
    batch_embed_times = []
    batch_extract_times = []
    
    print("  测试批处理时间...")
    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        cover_vec = torch.tensor(test_data[start_idx:end_idx], dtype=torch.float32, device=device)
        
        # 测量嵌入时间
        start_time = time.time()
        stego_vec, message = model.encode(cover_vec, random_msg=True)
        batch_time = time.time() - start_time
        batch_embed_times.append(batch_time / (end_idx - start_idx))  # 平均到每个样本
        
        # 测量提取时间
        start_time = time.time()
        _ = model.decode(stego_vec)
        batch_time = time.time() - start_time
        batch_extract_times.append(batch_time / (end_idx - start_idx))  # 平均到每个样本
    
    # 计算统计结果
    results = {
        "single_sample_embed_time_ms": float(np.mean(single_embed_times) * 1000),
        "single_sample_extract_time_ms": float(np.mean(single_extract_times) * 1000),
        "batch_embed_time_per_sample_ms": float(np.mean(batch_embed_times) * 1000),
        "batch_extract_time_per_sample_ms": float(np.mean(batch_extract_times) * 1000),
    }
    
    return results


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化水印模型
    print(f"初始化水印模型，使用预训练权重: {args.model_path}")
    watermark_model = VectorWatermark(
        vec_dim=args.vec_dim,
        msg_len=args.msg_len,
        model_path=args.model_path,
        device=args.device
    )
    
    # 加载测试数据
    test_data = load_test_data(args.data_path, args.test_samples)
    
    # 运行测试
    test_results = {}
    
    # 1. 评估嵌入质量（余弦相似度）
    test_results["cosine_similarity"] = evaluate_cosine_similarity(
        watermark_model, test_data, args.batch_size
    )
    
    # 2. 评估水印提取准确率
    test_results["accuracy"] = evaluate_accuracy(
        watermark_model, test_data, args.batch_size, args.bit_error_threshold
    )
    
    # 3. 评估CRC校验成功率
    test_results["crc_verification"] = evaluate_crc_verification(
        watermark_model, test_data, args.batch_size
    )
    
    # 4. 评估噪声鲁棒性
    test_results["noise_robustness"] = evaluate_noise_robustness(
        watermark_model, test_data, args.batch_size, args.bit_error_threshold
    )
    
    # 5. 评估时间性能
    test_results["performance"] = evaluate_performance(
        watermark_model, test_data, args.batch_size
    )
    
    # 打印测试结果摘要
    print("\n" + "="*50)
    print("测试结果摘要")
    print("="*50)
    
    # 1. 余弦相似度
    print("\n1. 嵌入质量（余弦相似度）:")
    for key, value in test_results["cosine_similarity"].items():
        print(f"  {key}: {value:.6f}")
    
    # 2. 提取准确率
    print("\n2. 水印提取准确率:")
    print(f"  比特错误率: {test_results['accuracy']['bit_error_rate']:.6f}")
    print(f"  容错阈值 {args.bit_error_threshold} 比特的准确率: {test_results['accuracy']['accuracy']:.2%}")
    
    # 3. CRC校验成功率
    print("\n3. CRC校验成功率:")
    print(f"  成功率: {test_results['crc_verification']['crc_verification_rate']:.2%}")
    
    # 4. 噪声鲁棒性
    print("\n4. 噪声鲁棒性:")
    noise_types = set()
    for key in test_results["noise_robustness"].keys():
        noise_name = key.split("_accuracy")[0] if "_accuracy" in key else key.split("_bit")[0]
        if "_crc" in key:
            noise_name = key.split("_crc")[0]
        noise_types.add(noise_name)
    
    for noise in sorted(noise_types):
        ber_key = f"{noise}_bit_error_rate"
        acc_key = f"{noise}_accuracy"
        crc_key = f"{noise}_crc_verification_rate"
        print(f"\n  噪声类型: {noise}")
        print(f"    比特错误率: {test_results['noise_robustness'][ber_key]:.6f}")
        print(f"    容错阈值 {args.bit_error_threshold} 比特的准确率: {test_results['noise_robustness'][acc_key]:.2%}")
        print(f"    CRC校验成功率: {test_results['noise_robustness'][crc_key]:.2%}")
    
    # 5. 时间性能
    print("\n5. 时间性能:")
    for key, value in test_results["performance"].items():
        if "embed" in key:
            print(f"  嵌入时间 ({key}): {value:.3f} 毫秒")
        else:
            print(f"  提取时间 ({key}): {value:.3f} 毫秒")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()