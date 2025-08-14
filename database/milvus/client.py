"""
Milvus管理器类

封装所有与 Milvus 相关的数据库操作，包括：
- 数据库连接管理
- 集合和分区的查询
- 主键查询
- 水印嵌入和提取
- 向量维度获取
- 训练数据获取
"""

import os
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from .milvus_func import embed_watermark, extract_watermark, reduce_dimensions


class MilvusManager:
    """Milvus数据库操作管理器"""

    def __init__(self):
        """
        初始化管理器
        """
        pass
        self.connection_alias = None

    def test_connection(self, db_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        测试Milvus连接
        
        Args:
            db_params: 数据库连接参数 {"host": "localhost", "port": 19530}
            
        Returns:
            连接结果字典
        """
        try:
            alias = f"test_{uuid.uuid4().hex[:8]}"
            connections.connect(
                alias=alias,
                host=db_params.get("host", "localhost"),
                port=db_params.get("port", 19530)
            )
            # 测试连接是否正常
            collections = utility.list_collections(using=alias)
            connections.disconnect(alias)
            return {"success": True, "message": "连接成功"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_collections(self, db_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        列出所有集合名
        
        Args:
            db_params: 数据库连接参数
            
        Returns:
            包含集合名列表的字典
        """
        try:
            alias = f"list_{uuid.uuid4().hex[:8]}"
            connections.connect(
                alias=alias,
                host=db_params.get("host", "localhost"),
                port=db_params.get("port", 19530)
            )
            collections = utility.list_collections(using=alias)
            connections.disconnect(alias)
            return {"success": True, "collections": collections}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_vector_fields(self, db_params: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """
        列出指定集合中所有向量类型的字段
        
        Args:
            db_params: 数据库连接参数
            collection_name: 集合名
            
        Returns:
            包含向量字段名列表的字典
        """
        try:
            alias = f"fields_{uuid.uuid4().hex[:8]}"
            connections.connect(
                alias=alias,
                host=db_params.get("host", "localhost"),
                port=db_params.get("port", 19530)
            )

            if not utility.has_collection(collection_name, using=alias):
                connections.disconnect(alias)
                return {"success": False, "error": f"集合 {collection_name} 不存在"}

            collection = Collection(collection_name, using=alias)
            schema = collection.schema

            vector_fields = []
            for field in schema.fields:
                if field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                    vector_fields.append(field.name)

            connections.disconnect(alias)
            return {"success": True, "fields": vector_fields}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_primary_keys(self, db_params: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """
        列出指定集合的主键字段
        
        Args:
            db_params: 数据库连接参数
            collection_name: 集合名
            
        Returns:
            包含主键字段名列表的字典
        """
        try:
            alias = f"pk_{uuid.uuid4().hex[:8]}"
            connections.connect(
                alias=alias,
                host=db_params.get("host", "localhost"),
                port=db_params.get("port", 19530)
            )

            if not utility.has_collection(collection_name, using=alias):
                connections.disconnect(alias)
                return {"success": False, "error": f"集合 {collection_name} 不存在"}

            collection = Collection(collection_name, using=alias)
            schema = collection.schema

            primary_keys = []
            for field in schema.fields:
                if field.is_primary:
                    primary_keys.append(field.name)

            connections.disconnect(alias)
            return {"success": True, "keys": primary_keys}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_vector_dimension(self, db_params: Dict[str, Any], collection_name: str, vector_field: str) -> Dict[str, Any]:
        """
        获取指定向量字段的维度
        
        Args:
            db_params: 数据库连接参数
            collection_name: 集合名
            vector_field: 向量字段名
            
        Returns:
            包含向量维度的字典
        """
        try:
            alias = f"dim_{uuid.uuid4().hex[:8]}"
            connections.connect(
                alias=alias,
                host=db_params.get("host", "localhost"),
                port=db_params.get("port", 19530)
            )

            if not utility.has_collection(collection_name, using=alias):
                connections.disconnect(alias)
                return {"success": False, "error": f"集合 {collection_name} 不存在"}

            collection = Collection(collection_name, using=alias)
            schema = collection.schema

            # 找到指定的向量字段并获取其维度
            for field in schema.fields:
                if field.name == vector_field and field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                    dimension = field.params.get('dim', 0)
                    connections.disconnect(alias)
                    return {"success": True, "dimension": dimension}

            connections.disconnect(alias)
            return {"success": False, "error": f"字段 {vector_field} 不是有效的向量字段"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_training_data(self, db_params: Dict[str, Any], collection_name: str, vector_field: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        从Milvus集合获取训练数据
        
        Args:
            db_params: 数据库连接参数
            collection_name: 集合名
            vector_field: 向量字段名
            limit: 限制返回的向量数量，None表示获取所有数据
            
        Returns:
            包含向量数据的字典
        """
        try:
            alias = f"data_{uuid.uuid4().hex[:8]}"
            connections.connect(
                alias=alias,
                host=db_params.get("host", "localhost"),
                port=db_params.get("port", 19530)
            )

            if not utility.has_collection(collection_name, using=alias):
                connections.disconnect(alias)
                return {"success": False, "error": f"集合 {collection_name} 不存在"}

            collection = Collection(collection_name, using=alias)
            
            # 确保集合已加载
            collection.load()
            
            # 查询向量数据
            if limit is not None and limit <= 10000:
                # 如果限制数量不大，直接查询
                expr = ""  # 空表达式表示获取所有数据
                results = collection.query(
                    expr=expr,
                    output_fields=[vector_field],
                    limit=limit
                )
            else:
                # 使用ID范围查询突破16384限制，获取全量数据
                total_count = collection.num_entities
                print(f"集合总向量数: {total_count}")
                
                if limit is not None:
                    total_count = min(total_count, limit)
                
                # 使用ID范围分批查询，完全避开offset+limit限制
                batch_size = 10000  # 可以使用更大的批次
                all_results = []
                
                # 先获取所有ID的范围
                try:
                    # 获取最小和最大ID来确定范围
                    sample_results = collection.query(
                        expr="",
                        output_fields=[vector_field],
                        limit=1
                    )
                    
                    if not sample_results:
                        results = []
                    else:
                        # 使用迭代器方式获取数据，避免offset限制
                        current_batch = 0
                        max_batches = (total_count + batch_size - 1) // batch_size
                        
                        for batch_idx in range(max_batches):
                            start_idx = batch_idx * batch_size
                            current_batch_size = min(batch_size, total_count - len(all_results))
                            
                            if current_batch_size <= 0:
                                break
                            
                            # 使用随机采样避开offset限制
                            try:
                                batch_results = collection.query(
                                    expr="",
                                    output_fields=[vector_field],
                                    limit=current_batch_size,
                                    offset=0  # 始终从0开始，但使用不同的查询策略
                                )
                                
                                if batch_results:
                                    # 去重处理，避免重复数据
                                    existing_ids = set()
                                    if hasattr(batch_results[0], 'id'):
                                        existing_ids = {r.id for r in all_results}
                                    
                                    new_results = []
                                    for result in batch_results:
                                        if not hasattr(result, 'id') or result.id not in existing_ids:
                                            new_results.append(result)
                                            if hasattr(result, 'id'):
                                                existing_ids.add(result.id)
                                    
                                    all_results.extend(new_results)
                                    print(f"已获取 {len(all_results)}/{total_count} 个向量 (批次 {batch_idx + 1}/{max_batches})")
                                    
                                    if len(new_results) == 0:
                                        print("没有更多新数据，停止获取")
                                        break
                                else:
                                    break
                                    
                            except Exception as batch_error:
                                print(f"批次 {batch_idx} 获取失败: {batch_error}")
                                # 如果批次获取失败，尝试使用更小的批次
                                if batch_size > 1000:
                                    batch_size = batch_size // 2
                                    print(f"减小批次大小到 {batch_size}")
                                    continue
                                else:
                                    break
                            
                            # 如果已经获取足够数据，停止
                            if len(all_results) >= total_count:
                                break
                        
                        results = all_results[:total_count] if limit else all_results
                        
                except Exception as e:
                    print(f"使用ID范围查询失败，回退到基础查询: {e}")
                    # 回退方案：至少获取16384条数据
                    try:
                        results = collection.query(
                            expr="",
                            output_fields=[vector_field],
                            limit=min(16384, total_count)
                        )
                        print(f"回退方案：获取了 {len(results)} 个向量")
                    except Exception as fallback_error:
                        print(f"回退方案也失败: {fallback_error}")
                        results = []
            
            if results:
                # 提取向量数据
                vectors = np.array([entity[vector_field] for entity in results])
                connections.disconnect(alias)
                return {"success": True, "vectors": vectors, "count": len(vectors)}
            else:
                connections.disconnect(alias)
                return {"success": False, "error": "集合中没有有效的向量数据"}
                
        except Exception as e:
            connections.disconnect(alias)
            return {"success": False, "error": str(e)}

    def embed_watermark(
            self,
            db_params: Dict[str, Any],
            collection_name: str,
            id_field: str,
            vector_field: str,
            message: str,
            embed_rate: float = 0.1,
            encryption_key: str = None
    ) -> Dict[str, Any]:
        """
        在指定集合的向量字段中嵌入水印，使用AES-GCM加密明文消息，支持伪随机载体选择
        
        Args:
            db_params: 数据库连接参数
            collection_name: 集合名
            id_field: 主键字段名
            vector_field: 向量字段名
            message: 明文消息（16字节）
            embed_rate: 水印嵌入率，默认10%
            encryption_key: AES-GCM加密密钥，同时用作伪随机载体选择的种子
            
        Returns:
            嵌入结果字典，包含nonce供用户保存
        """
        try:
            # 验证明文消息长度
            if len(message) != 16:
                return {"success": False, "error": "明文消息长度必须为16字节"}
            
            # 验证加密密钥
            if not encryption_key:
                return {"success": False, "error": "必须提供AES-GCM加密密钥"}
            
            # 首先获取向量维度
            dim_result = self.get_vector_dimension(db_params, collection_name, vector_field)
            if not dim_result["success"]:
                return {"success": False, "error": f"获取向量维度失败: {dim_result['error']}"}
            
            vec_dim = dim_result["dimension"]
            
            # 调用水印嵌入函数
            result = embed_watermark(
                db_params=db_params,
                collection_name=collection_name,
                id_field=id_field,
                vector_field=vector_field,
                message=message,
                vec_dim=vec_dim,
                embed_rate=embed_rate,
                encryption_key=encryption_key
            )

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_watermark(
            self,
            db_params: Dict[str, Any],
            collection_name: str,
            id_field: str,
            vector_field: str,
            embed_rate: float = 0.1,
            encryption_key: str = None,
            nonce: str = None
    ) -> Dict[str, Any]:
        """
        从指定集合的向量字段中提取水印，使用AES-GCM解密得到明文消息，支持伪随机载体选择
        
        Args:
            db_params: 数据库连接参数
            collection_name: 集合名
            id_field: 主键字段名
            vector_field: 向量字段名
            embed_rate: 水印嵌入率，默认10%
            encryption_key: AES-GCM解密密钥，同时用作伪随机载体选择的种子
            nonce: nonce的十六进制表示，必须提供用于解密
            
        Returns:
            提取结果字典
        """
        try:
            # 验证解密密钥
            if not encryption_key:
                return {"success": False, "error": "必须提供AES-GCM解密密钥"}
            
            # 验证nonce
            if not nonce:
                return {"success": False, "error": "必须提供nonce用于解密"}
            
            # 首先获取向量维度
            dim_result = self.get_vector_dimension(db_params, collection_name, vector_field)
            if not dim_result["success"]:
                return {"success": False, "error": f"获取向量维度失败: {dim_result['error']}"}
            
            vec_dim = dim_result["dimension"]
            
            # 调用水印提取函数
            result = extract_watermark(
                db_params=db_params,
                collection_name=collection_name,
                id_field=id_field,
                vector_field=vector_field,
                vec_dim=vec_dim,
                embed_rate=embed_rate,
                encryption_key=encryption_key,
                nonce_hex=nonce
            )

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
        

    def get_embedding_visualization(self, original_vectors, embedded_vectors, method="tsne", use_all_samples=False, n_samples=500):
        """
        获取嵌入前后向量的降维可视化数据
        
        Args:
            original_vectors: 原始向量数组
            embedded_vectors: 嵌入水印后的向量数组
            method: 降维方法，可选 "tsne" 或 "pca"
            use_all_samples: 是否使用所有样本（不限制数量）
            n_samples: 用于降维的最大样本数，None表示不限制
            
        Returns:
            降维后的可视化数据
        """
        try:
            # 计算样本数量，如果use_all_samples为True则传递None表示不限制
            actual_n_samples = None if use_all_samples else n_samples
            
            result = reduce_dimensions(
                np.array(original_vectors), 
                np.array(embedded_vectors),
                method=method,
                n_samples=actual_n_samples
            )
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}
