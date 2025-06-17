#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vector_watermark.py - 向量水印编码/解码实现
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict, Optional

from src.vector_encoder import AdvVectorEncoder
from src.vector_decoder import AdvVectorDecoder


class VectorWatermark:
    """向量水印处理类：加载预训练模型，提供编码和解码功能"""
    
    def __init__(
        self,
        vec_dim: int = 384,
        msg_len: int = 96,
        model_path: Optional[str] = None,
        device: str = None
    ):
        """
        初始化向量水印处理器
        
        参数:
            vec_dim: 向量维度
            msg_len: 消息长度（比特数）
            model_path: 预训练模型权重路径
            device: 计算设备，None 时自动选择
        """
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.vec_dim = vec_dim
        self.msg_len = msg_len
        
        # 初始化模型
        self.encoder = AdvVectorEncoder(vec_dim, msg_len)
        self.decoder = AdvVectorDecoder(vec_dim, msg_len)
        
        # 将模型移至指定设备
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        # 如果提供了模型路径，则加载权重
        if model_path:
            self.load_model(model_path)
        
        # 默认评估模式
        self.encoder.eval()
        self.decoder.eval()
    
    def load_model(self, model_path: str) -> None:
        """
        加载预训练模型权重
        
        参数:
            model_path: 模型权重文件路径
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
            
        # 在load_model方法中修改:
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # 检查预期的键
        if 'enc' not in checkpoint or 'dec' not in checkpoint:
            raise ValueError(f"模型文件格式错误，缺少'enc'或'dec'键")
        
        self.encoder.load_state_dict(checkpoint['enc'])
        self.decoder.load_state_dict(checkpoint['dec'])
        print(f"成功加载模型权重: {model_path}")
        
    def encode(
        self, 
        cover_vec: Union[torch.Tensor, np.ndarray], 
        message: Union[torch.Tensor, np.ndarray, None] = None,
        random_msg: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将消息编码到载体向量中
        
        参数:
            cover_vec: 载体向量，形状 (batch_size, vec_dim) 
            message: 待编码消息，形状 (batch_size, msg_len)，值为0/1
                     如果为None且random_msg=True，则生成随机消息
            random_msg: 是否生成随机消息
            
        返回:
            Tuple[隐写向量, 编码的消息]
        """
        # 转换输入类型
        if isinstance(cover_vec, np.ndarray):
            cover_vec = torch.from_numpy(cover_vec)
        
        # 确保输入是浮点类型
        cover_vec = cover_vec.float().to(self.device)
        
        # 处理输入维度
        if cover_vec.dim() == 1:
            cover_vec = cover_vec.unsqueeze(0)  # 添加批次维度
            
        batch_size = cover_vec.shape[0]
            
        # 处理消息
        if message is None:
            if random_msg:
                # 生成随机二进制消息
                message = torch.randint(0, 2, (batch_size, self.msg_len), 
                                       device=self.device).float()
            else:
                raise ValueError("必须提供消息或设置 random_msg=True")
        elif isinstance(message, np.ndarray):
            message = torch.from_numpy(message).float().to(self.device)
        else:
            message = message.float().to(self.device)
            
        # 确保消息形状正确
        if message.dim() == 1:
            message = message.unsqueeze(0)
            
        # 编码过程
        with torch.no_grad():
            stego_vec = self.encoder(cover_vec, message)
            
        return stego_vec, message
    
    def decode(
        self, 
        stego_vec: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        从隐写向量中解码消息
        
        参数:
            stego_vec: 隐写向量，形状 (batch_size, vec_dim)
            
        返回:
            提取的消息，形状为 (batch_size, msg_len)，值为0/1
        """
        # 转换输入类型
        if isinstance(stego_vec, np.ndarray):
            stego_vec = torch.from_numpy(stego_vec)
            
        # 确保输入是浮点类型
        stego_vec = stego_vec.float().to(self.device)
        
        # 处理输入维度
        if stego_vec.dim() == 1:
            stego_vec = stego_vec.unsqueeze(0)  # 添加批次维度
            
        # 解码过程
        with torch.no_grad():
            logits = self.decoder(stego_vec)
            message = torch.sigmoid(logits) > 0.5
            
        return message.float()
    
    def compute_ber(
        self, 
        original_msg: torch.Tensor, 
        decoded_msg: torch.Tensor
    ) -> float:
        """
        计算比特错误率 (BER)
        
        参数:
            original_msg: 原始消息
            decoded_msg: 解码的消息
            
        返回:
            比特错误率，取值范围 [0, 1]
        """
        if original_msg.shape != decoded_msg.shape:
            raise ValueError(f"消息形状不匹配: {original_msg.shape} vs {decoded_msg.shape}")
            
        # 计算不匹配的比特占比
        ber = (decoded_msg != original_msg).float().mean().item()
        return ber
    
    def watermark_vector(
        self, 
        cover_vec: Union[torch.Tensor, np.ndarray],
        message: Union[torch.Tensor, np.ndarray, None] = None,
        return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        完整的水印添加过程
        
        参数:
            cover_vec: 载体向量
            message: 水印消息，如果为None则随机生成
            return_numpy: 是否返回numpy数组
            
        返回:
            隐写向量
        """
        stego_vec, _ = self.encode(cover_vec, message, random_msg=(message is None))
        
        if return_numpy and isinstance(stego_vec, torch.Tensor):
            return stego_vec.cpu().numpy()
        return stego_vec
    
    def extract_watermark(
        self, 
        stego_vec: Union[torch.Tensor, np.ndarray],
        return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        从水印向量中提取消息
        
        参数:
            stego_vec: 隐写向量
            return_numpy: 是否返回numpy数组
            
        返回:
            提取的消息
        """
        message = self.decode(stego_vec)
        
        if return_numpy and isinstance(message, torch.Tensor):
            return message.cpu().numpy()
        return message


# 使用示例
if __name__ == "__main__":
    # 创建水印处理器实例
    watermark = VectorWatermark(
        vec_dim=384,
        msg_len=96,
        model_path="results/vector_val/best.pt"
    )
    
    # 示例向量和消息
    cover = torch.randn(10, 384)  # 10个模拟的向量
    msg = torch.randint(0, 2, (10, 96)).float()  # 随机二进制消息
    
    # 编码
    stego, original_msg = watermark.encode(cover, msg)
    print(f"载体向量形状: {cover.shape}")
    print(f"隐写向量形状: {stego.shape}")
    
    # 计算失真
    distortion = F.mse_loss(cover.to(watermark.device), stego).item()
    print(f"失真 (MSE): {distortion:.6f}")
    
    # 解码
    extracted_msg = watermark.decode(stego)
    
    # 评估
    ber = watermark.compute_ber(original_msg, extracted_msg)
    print(f"比特错误率: {ber:.2%}")