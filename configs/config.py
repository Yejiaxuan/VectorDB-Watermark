"""
向量水印系统统一配置文件

包含训练、水印、数据库等核心参数的统一管理
"""
import os


class Config:
    """统一配置类"""

    # ==================== HNSW参数 ====================
    HNSW_M = 16  # HNSW M参数 (连接数)
    HNSW_EF_CONSTRUCTION = 200  # HNSW构建时的ef参数
    HNSW_EF_SEARCH = 50  # HNSW搜索时的ef参数

    # ==================== 水印参数 ====================
    MSG_LEN = 24  # 消息长度 (4位索引 + 4位CRC + 16位载荷)
    BLOCK_PAYLOAD = 16  # 每块载荷比特数
    BLOCK_COUNT = 16  # 块数量
    DEFAULT_EMBED_RATE = 0.1  # 默认水印嵌入率 (10%)
    BIT_ERROR_THRESHOLD = 1  # 容错比特错误阈值
    CRC_POLYNOMIAL = 0x3  # CRC-4多项式

    # ==================== 训练参数 ====================
    BATCH_SIZE = 8192  # 批处理大小
    EPOCHS = 100  # 训练轮数
    LEARNING_RATE = 3e-4  # 学习率
    VAL_RATIO = 0.15  # 验证集比例
    SEED = 42  # 随机种子

    # ==================== 测试参数 ====================
    TEST_SAMPLES = 1000  # 测试样本数量
    TEST_BATCH_SIZE = 128  # 测试批处理大小

    # ==================== 路径配置 ====================
    RESULTS_DIR = "algorithms/deep_learning/results"

    @staticmethod
    def get_model_path(vec_dim: int) -> str:
        """
        根据向量维度生成模型路径
        
        Args:
            vec_dim: 向量维度
            
        Returns:
            模型文件路径
        """
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            f'algorithms/deep_learning/results/vector_{vec_dim}d/best.pt'
        )
    
    @staticmethod
    def get_results_dir(vec_dim: int) -> str:
        """
        根据向量维度生成结果目录路径
        
        Args:
            vec_dim: 向量维度
            
        Returns:
            结果目录路径
        """
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            f'algorithms/deep_learning/results/vector_{vec_dim}d'
        )


config = Config()
