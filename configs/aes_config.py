"""
AES-GCM加密水印系统配置文件

定义与AES-GCM加密相关的参数配置
"""

class AESConfig:
    # AES-GCM加密参数
    PLAINTEXT_LENGTH = 16  # 明文消息长度（字节）
    CIPHERTEXT_LENGTH = 32  # 密文总长度（16字节密文 + 16字节标签）
    
    # 密钥派生参数
    PBKDF2_ITERATIONS = 100000  # PBKDF2迭代次数
    SALT = b'DbWM_Salt_2024'  # 固定盐值
    
    # 向量水印相关参数（保持兼容）
    MSG_LEN = 256  # 消息总比特长度 (32字节 * 8)
    BLOCK_PAYLOAD = 128  # 每块载荷比特数 (16字节 * 8)  
    BLOCK_COUNT = 16  # 总块数 (32字节 / 2字节per块)
    
    @staticmethod
    def validate_plaintext(message: str) -> bool:
        """验证明文消息格式"""
        return len(message) == AESConfig.PLAINTEXT_LENGTH
    
    @staticmethod
    def validate_encryption_key(key: str) -> bool:
        """验证加密密钥格式"""
        return len(key) > 0  # 基本验证，实际密钥强度由PBKDF2保证
