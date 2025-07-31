import torch
import numpy as np
from typing import Tuple, List, Union, Optional
import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from algorithms.deep_learning.watermark import VectorWatermark

class DimensionShuffleAttack:
    """向量维度混淆攻击类"""
    
    def __init__(self, seed: int = 42, perturb_ratio: float = 1.0):
        """
        初始化攻击
        
        参数:
            seed: 随机种子，用于生成维度置换
            perturb_ratio: 需要扰动的维度比例 (0.0-1.0)
        """
        self.seed = seed
        self.perturb_ratio = max(0.0, min(1.0, perturb_ratio))  # 确保范围在0到1之间
        self.permutation = None
        self.inverse_perm = None
        self.perturbed_dims = None
        
    def generate_permutation(self, dim: int) -> np.ndarray:
        """
        生成维度置换
        
        参数:
            dim: 向量维度
            
        返回:
            维度置换数组
        """
        rng = np.random.RandomState(self.seed)
        
        # 计算需要置换的维度数量
        num_perturbed_dims = int(dim * self.perturb_ratio)
        
        # 创建初始置换（等同于原始顺序）
        self.permutation = np.arange(dim)
        
        # 如果需要扰动的维度比例大于0
        if num_perturbed_dims > 0:
            # 随机选择要扰动的维度
            self.perturbed_dims = rng.choice(dim, num_perturbed_dims, replace=False)
            
            # 仅对选择的维度进行置换
            perm_subset = rng.permutation(num_perturbed_dims)
            selected_dims = self.perturbed_dims.copy()
            
            # 应用子集置换
            for i, idx in enumerate(self.perturbed_dims):
                self.permutation[idx] = selected_dims[perm_subset[i]]
        
        # 计算逆置换
        self.inverse_perm = np.argsort(self.permutation)
        
        return self.permutation
    
    def attack(self, vectors: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        对向量进行维度混淆攻击
        
        参数:
            vectors: 输入向量，形状为(batch_size, vec_dim)
            
        返回:
            攻击后的向量
        """
        is_torch = isinstance(vectors, torch.Tensor)
        if is_torch:
            vectors_np = vectors.detach().cpu().numpy()
        else:
            vectors_np = vectors
            
        # 确保输入是二维的
        if vectors_np.ndim == 1:
            vectors_np = vectors_np.reshape(1, -1)
            
        # 生成或使用已有置换
        if self.permutation is None or len(self.permutation) != vectors_np.shape[1]:
            self.generate_permutation(vectors_np.shape[1])
            
        # 应用维度置换
        attacked_vectors = vectors_np.copy()
        for i in range(vectors_np.shape[0]):
            attacked_vectors[i] = vectors_np[i, self.permutation]
        
        # 返回与输入相同类型的结果
        if is_torch:
            return torch.tensor(attacked_vectors, device=vectors.device, dtype=vectors.dtype)
        return attacked_vectors


def load_data_from_npy(file_path: str, vec_dim: int = None) -> torch.Tensor:
    """
    从npy文件加载向量数据
    
    参数:
        file_path: npy文件路径
        vec_dim: 期望的向量维度，如果不为None则会验证数据维度
        
    返回:
        加载的向量数据张量
    """
    try:
        data = np.load(file_path)
        
        # 确保数据是二维的
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # 验证维度
        if vec_dim is not None and data.shape[1] != vec_dim:
            print(f"警告: 加载的数据维度 ({data.shape[1]}) 与期望维度 ({vec_dim}) 不匹配!")
            
        return torch.tensor(data, dtype=torch.float32)
    except Exception as e:
        print(f"加载数据出错: {str(e)}")
        return None


def evaluate_attack(
    vec_dim: int,
    msg_len: int,
    model_path: str,
    data_path: str = None,
    num_samples: int = 100,
    seed_range: Tuple[int, int] = (0, 10),
    perturb_ratios: List[float] = [1.0]
) -> dict:
    """
    评估不同种子和扰动比例下的维度混淆攻击效果
    
    参数:
        vec_dim: 向量维度
        msg_len: 消息长度
        model_path: 水印模型路径
        data_path: npy数据文件路径，如果为None则生成随机数据
        num_samples: 测试样本数量 (当data_path为None时使用)
        seed_range: 随机种子范围
        perturb_ratios: 扰动比例列表
        
    返回:
        包含不同扰动比例下解码成功率的字典
    """
    # 加载水印模型
    watermarker = VectorWatermark(vec_dim, msg_len, model_path=model_path)
    
    # 加载或生成测试向量
    if data_path and os.path.exists(data_path):
        print(f"从文件加载数据: {data_path}")
        vectors = load_data_from_npy(data_path, vec_dim)
        if vectors is None:
            print("数据加载失败，使用随机生成的数据替代")
            vectors = torch.randn(num_samples, vec_dim)
        else:
            num_samples = min(vectors.shape[0], num_samples)
            vectors = vectors[:num_samples]
    else:
        print(f"生成随机测试向量: {num_samples} 个样本")
        vectors = torch.randn(num_samples, vec_dim)
    
    # 生成并嵌入随机水印消息
    watermarked_vectors, original_messages = watermarker.encode(vectors, random_msg=True)
    
    # 评估不同扰动比例和种子的攻击效果
    results = {}
    seeds = list(range(seed_range[0], seed_range[1]))
    
    for ratio in perturb_ratios:
        print(f"\n评估扰动比例: {ratio:.2f}")
        success_rates = []
        
        for seed in seeds:
            # 初始化攻击
            attack = DimensionShuffleAttack(seed=seed, perturb_ratio=ratio)
            
            # 执行攻击
            attacked_vectors = attack.attack(watermarked_vectors)
            
            # 尝试解码
            decoded_messages = watermarker.decode(attacked_vectors)
            
            # 计算比特准确率
            bit_accuracy = (decoded_messages == original_messages).float().mean(dim=1)
            
            # 计算消息完全正确率（所有比特都正确）
            perfect_matches = (bit_accuracy == 1.0).float().mean().item()
            success_rates.append(perfect_matches * 100)  # 转为百分比
            
            print(f"  种子 {seed}: 解码成功率 = {perfect_matches:.2%}")
            
        results[ratio] = success_rates
    
    return results, seeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="向量维度混淆攻击评估")
    parser.add_argument("--dim", type=int, default=384, help="向量维度")
    parser.add_argument("--msg_len", type=int, default=24, help="消息长度")
    parser.add_argument("--model_path", type=str, default=None, help="水印模型路径")
    parser.add_argument("--data_path", type=str, default=None, help="包含向量数据的npy文件路径")
    parser.add_argument("--num_samples", type=int, default=10000, help="测试样本数量")
    parser.add_argument("--seed_start", type=int, default=0, help="种子范围起始")
    parser.add_argument("--seed_end", type=int, default=10, help="种子范围结束")
    parser.add_argument("--perturb_ratios", type=float, nargs='+', default=[0.0, 0.1, 0.3, 0.5, 0.8, 1.0], 
                        help="扰动比例列表 (0.0-1.0)")
    args = parser.parse_args()
    
    # 如果未指定模型路径，使用默认路径
    if args.model_path is None:
        model_dir = Path(__file__).parent / "results" / f"vector_{args.dim}d"
        args.model_path = str(model_dir / "best.pt")

    if args.data_path is None:
        data_dir = Path(__file__).parent / "dataset" / f"qa"
        args.data_path = str(data_dir / "nq_qa_combined_384d.npy")

    print(f"评估维度混淆攻击 (向量维度: {args.dim})")
    print(f"使用模型: {args.model_path}")
    print(f"测试样本: {args.num_samples}")
    print(f"种子范围: {args.seed_start} - {args.seed_end}")
    print(f"扰动比例: {args.perturb_ratios}")
    
    # 运行评估
    seeds = list(range(args.seed_start, args.seed_end))
    results, seeds = evaluate_attack(
        args.dim, 
        args.msg_len,
        args.model_path,
        args.data_path,
        args.num_samples,
        (args.seed_start, args.seed_end),
        args.perturb_ratios
    )
    