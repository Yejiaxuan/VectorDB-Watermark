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
from .milvus_func import embed_watermark, extract_watermark


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
                # 如果没有限制或限制很大，使用分批查询避免Milvus查询窗口限制
                total_count = collection.num_entities
                print(f"集合总向量数: {total_count}")
                
                if limit is not None:
                    total_count = min(total_count, limit)
                
                # 使用分批查询，避免offset+limit超过16384的限制
                batch_size = 5000  # 保守的批次大小
                all_results = []
                
                # 通过offset分批查询
                current_offset = 0
                
                while len(all_results) < total_count:
                    remaining = total_count - len(all_results)
                    current_batch_size = min(batch_size, remaining)
                    
                    # 确保offset+limit不超过16384
                    if current_offset + current_batch_size > 16384:
                        current_batch_size = 16384 - current_offset
                        if current_batch_size <= 0:
                            # 如果无法继续分批，跳出循环
                            print(f"达到Milvus查询窗口限制，已获取 {len(all_results)} 个向量")
                            break
                    
                    batch_results = collection.query(
                        expr="",
                        output_fields=[vector_field],
                        limit=current_batch_size,
                        offset=current_offset
                    )
                    
                    if not batch_results:
                        break
                        
                    all_results.extend(batch_results)
                    current_offset += len(batch_results)
                    
                    print(f"已获取 {len(all_results)}/{total_count} 个向量")
                    
                    # 如果这批数据少于请求的数量，说明数据已经全部获取
                    if len(batch_results) < current_batch_size:
                        break
                
                results = all_results
            
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
            embed_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        在指定集合的向量字段中嵌入水印，不生成ID文件
        
        Args:
            db_params: 数据库连接参数
            collection_name: 集合名
            id_field: 主键字段名
            vector_field: 向量字段名
            message: 水印消息
            embed_rate: 水印嵌入率，默认10%
            
        Returns:
            嵌入结果字典
        """
        try:
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
                ids_file=None  # 不生成ID文件
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
            embed_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        从指定集合的向量字段中提取水印，重新计算低入度节点
        
        Args:
            db_params: 数据库连接参数
            collection_name: 集合名
            id_field: 主键字段名
            vector_field: 向量字段名
            embed_rate: 水印嵌入率，默认10%
            
        Returns:
            提取结果字典
        """
        try:
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
                ids_file=None  # 不使用ID文件
            )

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
