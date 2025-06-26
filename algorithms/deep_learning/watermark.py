import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict, Optional

from .encoder import AdvVectorEncoder
from .decoder import AdvVectorDecoder


class VectorWatermark:
    """向量水印处理类：加载预训练模型，提供编码和解码功能"""
    
    def __init__(
        self,
        vec_dim: int = 384,
        msg_len: int = 24,  # 修改为24位消息长度，符合dataset格式
        model_path: Optional[str] = None,
        device: str = None
    ):
        """
        初始化向量水印处理器
        
        参数:
            vec_dim: 向量维度
            msg_len: 消息长度（比特数），默认24位（4位索引+4位CRC+16位载荷）
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
    
    def generate_message(self, batch_size: int = 1) -> torch.Tensor:
        """
        按照dataset格式生成消息：4位索引 + 4位CRC + 16位载荷
        
        参数:
            batch_size: 批量大小
            
        返回:
            消息张量，形状 (batch_size, 24)
        """
        messages = []
        
        for _ in range(batch_size):
            # 1) 随机选一个块索引 k ∈ [0,16)
            k = np.random.randint(0, 16)
            idx_bits = [(k >> i) & 1 for i in reversed(range(4))]  # MSB first
            
            # 2) 计算 4 bit CRC-4 校验（多项式 0x3）
            reg = 0
            for bit in idx_bits:
                reg ^= (bit << 3)
                for _ in range(4):
                    if reg & 0x8:
                        reg = ((reg << 1) & 0xF) ^ 0x3
                    else:
                        reg = (reg << 1) & 0xF
            crc_bits = [(reg >> i) & 1 for i in reversed(range(4))]
            
            # 3) 随机生成 16 bit payload
            payload = np.random.randint(0, 2, size=(16,), dtype=np.uint8).tolist()
            
            # 4) 拼成 24 bit 消息
            msg_bits = idx_bits + crc_bits + payload
            messages.append(msg_bits)
            
        return torch.tensor(messages, dtype=torch.float32, device=self.device)
        
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
                # 生成符合指定格式的随机消息
                message = self.generate_message(batch_size)
            else:
                raise ValueError("必须提供消息或设置 random_msg=True")
        elif isinstance(message, np.ndarray):
            message = torch.from_numpy(message).float().to(self.device)
        else:
            message = message.float().to(self.device)
            
        # 确保消息形状正确
        if message.dim() == 1:
            message = message.unsqueeze(0)
            
        # 存储原始向量的范数，用于后续恢复
        original_norms = torch.norm(cover_vec, p=2, dim=1, keepdim=True)
        
        # 对输入向量进行L2归一化
        normalized_cover = F.normalize(cover_vec, p=2, dim=1)
            
        # 编码过程
        with torch.no_grad():
            normalized_stego = self.encoder(normalized_cover, message)
            
        # 反归一化，恢复原始范数
        stego_vec = normalized_stego * original_norms
            
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
            
        # 对输入向量进行L2归一化
        normalized_stego = F.normalize(stego_vec, p=2, dim=1)
            
        # 解码过程
        with torch.no_grad():
            logits = self.decoder(normalized_stego)
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
            message: 水印消息，如果为None则随机生成符合格式的消息
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
    
    def verify_message(self, message: torch.Tensor) -> bool:
        """
        验证消息的CRC校验和是否正确
        
        参数:
            message: 形状为(msg_len,)或(batch_size, msg_len)的消息
            
        返回:
            校验是否通过的布尔值或布尔值列表
        """
        if message.dim() == 1:
            message = message.unsqueeze(0)
        
        batch_size = message.shape[0]
        results = []
        
        for i in range(batch_size):
            msg = message[i]
            idx_bits = msg[:4].cpu().int().tolist()
            crc_bits = msg[4:8].cpu().int().tolist()
            
            # 计算CRC校验
            reg = 0
            for bit in idx_bits:
                reg ^= (bit << 3)
                for _ in range(4):
                    if reg & 0x8:
                        reg = ((reg << 1) & 0xF) ^ 0x3
                    else:
                        reg = (reg << 1) & 0xF
            expected_crc = [(reg >> j) & 1 for j in reversed(range(4))]
            
            # 比较计算的CRC和解码的CRC
            is_valid = (expected_crc == crc_bits)
            results.append(is_valid)
            
        return results[0] if len(results) == 1 else results