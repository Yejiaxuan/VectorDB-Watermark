"""
vector_watermark_demo.py - 向量水印编码/解码演示程序
"""

import torch
import torch.nn.functional as F
import numpy as np
from .watermark import VectorWatermark
from .dataset import VectorWatermarkSet

# 创建水印处理器实例 - 更新为24位消息长度
watermark = VectorWatermark(
    vec_dim=384,
    msg_len=24,  # 更新为24位：4位索引+4位CRC+16位载荷
    model_path="algorithms/deep_learning/results/vector_val/best.pt"
)

# 加载训练时使用的同一数据集
data_path = "algorithms/deep_learning/dataset/msmarco/msmarco.npy"
dataset = VectorWatermarkSet(data_path, msg_len=24)  # 更新为24位消息长度

print("\n===== 基本水印功能测试 =====")

# 随机选择50个向量进行测试
indices = np.random.choice(len(dataset), 50, replace=False)
test_vectors = []
for idx in indices:
    cover, _ = dataset[idx]  # 忽略数据集中的消息，我们将使用自己的消息
    test_vectors.append(cover)

# 将向量堆叠成批次
cover_sample = torch.stack(test_vectors)
print(f"载体样本形状: {cover_sample.shape}")

# 测试 generate_message 功能
print("\n===== 测试随机消息生成 =====")
random_msg = watermark.generate_message(batch_size=50)
print(f"生成的随机消息形状: {random_msg.shape}")
print(f"随机消息示例:\n{random_msg[0].cpu().numpy()}")
print(f"索引位: {random_msg[0, :4].cpu().numpy()}")
print(f"CRC位: {random_msg[0, 4:8].cpu().numpy()}")
print(f"载荷: {random_msg[0, 8:].cpu().numpy()}")

# 嵌入随机生成的消息
print("\n===== 测试水印嵌入和提取 =====")
# 注意：现在不需要手动归一化，encode和decode函数内部会处理
stego, original_msg = watermark.encode(cover_sample[:50], random_msg)

print(f"载体向量形状: {cover_sample[:50].shape}")
print(f"隐写向量形状: {stego.shape}")

# 计算失真
distortion = F.mse_loss(cover_sample[:50].to(watermark.device), stego).item()
print(f"失真 (MSE): {distortion:.6f}")

# 解码
extracted_msg = watermark.decode(stego)

# 评估
ber = watermark.compute_ber(original_msg, extracted_msg)
print(f"比特错误率: {ber:.2%}")

# 测试消息验证功能
print("\n===== 测试消息验证 =====")
validation_results = watermark.verify_message(extracted_msg)
print(f"消息验证结果: {validation_results}")

# 测试水印向量和提取水印的快捷方法
print("\n===== 测试便捷包装方法 =====")
test_vector = cover_sample[10:11]  # 获取一个测试向量

# 使用便捷方法添加水印
watermarked_vec = watermark.watermark_vector(test_vector, return_numpy=False)
print(f"添加水印后向量形状: {watermarked_vec.shape}")

# 使用便捷方法提取水印
extracted_watermark = watermark.extract_watermark(watermarked_vec, return_numpy=False)
print(f"提取的水印形状: {extracted_watermark.shape}")
print(f"提取的水印: {extracted_watermark[0].cpu().numpy()}")

# 测试随机消息嵌入
print("\n===== 测试随机消息嵌入 =====")
watermarked_vec_random = watermark.watermark_vector(test_vector, message=None, return_numpy=False)
extracted_random = watermark.extract_watermark(watermarked_vec_random)
print(f"随机嵌入消息验证结果: {watermark.verify_message(extracted_random)}")

print("\n===== 诊断信息 =====")
print(f"使用设备: {watermark.device}")

# 1. 检查是否有任何编码/解码发生
print("\n1. 编码前后的向量差异:")
cover_sample_vec = cover_sample[0].cpu().numpy()
stego_sample_vec = stego[0].cpu().numpy()
diff = np.abs(cover_sample_vec - stego_sample_vec)
print(f"   最大差异: {diff.max():.6f}")
print(f"   平均差异: {diff.mean():.6f}")

# 2. 检查解码是否产生变化
print("\n2. 解码结果分布:")
# 获取原始解码输出（不经过阈值处理）
with torch.no_grad():
    # 对stego应用L2归一化后再输入解码器
    normalized_stego = F.normalize(stego, p=2, dim=1)
    raw_logits = watermark.decoder(normalized_stego)
    probabilities = torch.sigmoid(raw_logits).cpu().numpy()

# 检查概率分布
hist, _ = np.histogram(probabilities, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
print(f"   概率分布: {hist}")
print(f"   中位数概率: {np.median(probabilities):.4f}")
print(f"   最大概率: {np.max(probabilities):.4f}")
print(f"   最小概率: {np.min(probabilities):.4f}")

# 3. 对比原始和提取的消息
print("\n3. 消息对比:")
original_sample = original_msg[0].cpu().numpy()
extracted_sample = extracted_msg[0].cpu().numpy()
print(f"   原始消息: {original_sample}")
print(f"   提取消息: {extracted_sample}")
print(f"   匹配部分: {np.mean(original_sample == extracted_sample):.2%}")

# 4. 尝试不同阈值
print("\n4. 不同阈值的BER:")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    thresholded = (probabilities > threshold).astype(float)
    ber_t = np.mean(thresholded != original_msg.cpu().numpy())
    print(f"   阈值 {threshold:.1f}: BER = {ber_t:.2%}")

# 5. 长距离水印特性测试 - 随机扰动
print("\n5. 抗扰动测试:")
noise_levels = [0.0001, 0.001, 0.01, 0.05, 0.1]
for noise in noise_levels:
    # 添加高斯噪声
    noisy_stego = stego + torch.randn_like(stego) * noise
    # 解码
    noisy_extracted = watermark.decode(noisy_stego)
    # 计算BER
    noisy_ber = watermark.compute_ber(original_msg, noisy_extracted)
    # 验证消息
    noisy_valid = watermark.verify_message(noisy_extracted)
    print(f"   噪声水平 {noise:.5f}: BER = {noisy_ber:.2%}, 验证通过率: {np.mean(noisy_valid):.2%}")