"""
NQ数据集插入Milvus脚本

该脚本用于将nq_qa_combined_384d.npy数据文件插入到Milvus数据库中
创建名为'nq_qa_combined'的集合，包含id和embedding字段
"""

import os
import sys
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


def create_collection():
    """创建Milvus集合"""
    collection_name = "nq_qa_combined"
    
    # 如果集合已存在，先删除
    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在，正在删除...")
        utility.drop_collection(collection_name)
    
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    
    # 创建集合schema
    schema = CollectionSchema(fields, "NQ QA combined dataset with 384d embeddings")
    
    # 创建集合
    collection = Collection(collection_name, schema)
    print(f"成功创建集合: {collection_name}")
    
    return collection


def load_nq_data(file_path):
    """加载nq数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    print(f"正在加载数据文件: {file_path}")
    data = np.load(file_path)
    print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
    
    return data


def insert_data_in_batches(collection, data, batch_size=1000):
    """分批插入数据到Milvus"""
    total_rows = data.shape[0]
    inserted_count = 0
    
    print(f"开始插入数据，总计: {total_rows} 条记录")
    
    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        batch_data = data[i:end_idx]
        
        # 准备批量数据
        entities = [
            list(range(i, end_idx)),  # id字段：从0开始的连续整数
            batch_data.tolist()       # embedding字段：向量数据
        ]
        
        try:
            # 插入数据
            collection.insert(entities)
            inserted_count += len(batch_data)
            
            print(f"已插入 {inserted_count}/{total_rows} 条记录 ({inserted_count/total_rows*100:.1f}%)")
            
        except Exception as e:
            print(f"插入批次 {i}-{end_idx} 时出错: {e}")
            continue
    
    # 刷新数据确保持久化
    collection.flush()
    print("数据插入完成并已刷新到磁盘")
    
    return inserted_count


def create_index(collection):
    """为集合创建索引以加速搜索"""
    print("正在创建向量索引...")
    
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    
    collection.create_index("embedding", index_params)
    print("索引创建完成")


def verify_data(collection):
    """验证插入的数据"""
    collection.load()  # 加载集合到内存
    
    # 获取集合统计信息（使用Milvus的统计接口）
    print(f"集合中的记录数: {collection.num_entities}")
    
    # 查询前5条记录作为示例
    results = collection.query(
        expr="id >= 0",
        output_fields=["id", "embedding"],
        limit=5
    )
    
    print("前5条记录示例:")
    for result in results:
        vector_preview = result["embedding"][:10]  # 只显示前10个维度
        print(f"ID: {result['id']}, 向量预览: {vector_preview}")


def main():
    """主函数"""
    # 配置连接参数
    host = "localhost"
    port = 19530
    
    # 数据文件路径
    data_file = "database/milvus/docker/nq_qa_combined_384d.npy"
    
    try:
        # 连接到Milvus
        print(f"正在连接到Milvus服务器 {host}:{port}")
        connections.connect("default", host=host, port=port)
        print("连接成功")
        
        # 加载数据
        data = load_nq_data(data_file)
        
        # 创建集合
        collection = create_collection()
        
        # 插入数据
        inserted_count = insert_data_in_batches(collection, data, batch_size=1000)
        
        # 创建索引
        create_index(collection)
        
        # 验证数据
        verify_data(collection)
        
        print(f"\n✅ 数据插入完成！")
        print(f"   集合名称: nq_qa_combined")
        print(f"   插入记录数: {inserted_count}")
        print(f"   向量维度: 384")
        print(f"   主键字段: id")
        print(f"   向量字段: embedding")
        
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        sys.exit(1)
    
    finally:
        # 断开连接
        connections.disconnect("default")
        print("已断开Milvus连接")


if __name__ == "__main__":
    main() 