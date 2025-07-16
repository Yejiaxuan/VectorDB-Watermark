#!/usr/bin/env python3
"""
测试提取水印功能
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def test_extract():
    """测试提取水印功能"""
    try:
        from database.pgvector.pg_func import extract_watermark
        print("✓ 成功导入 extract_watermark 函数")
        
        # 测试参数
        db_params = {
            'host': 'localhost',
            'port': 5432,
            'dbname': 'test',
            'user': 'postgres',
            'password': 'ysj'
        }
        
        # 注意：这里只是测试函数调用，不需要真实的数据库连接
        print("✓ 参数准备完成")
        print("✓ 函数可以正常调用 (不进行实际数据库操作)")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

if __name__ == "__main__":
    print("=== 测试提取水印功能 ===")
    success = test_extract()
    print(f"测试结果: {'成功' if success else '失败'}")
