"""
Milvus管理器类

封装所有与 Milvus 相关的数据库操作，包括：
- 数据库连接管理
- 集合和分区的查询
- 主键查询
- 水印嵌入和提取
- 文件管理
"""

import os
import uuid
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
            # 调用水印嵌入函数（不生成ID文件）
            result = embed_watermark(
                db_params=db_params,
                collection_name=collection_name,
                id_field=id_field,
                vector_field=vector_field,
                message=message,
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
            # 调用水印提取函数（重新计算低入度节点）
            result = extract_watermark(
                db_params=db_params,
                collection_name=collection_name,
                id_field=id_field,
                vector_field=vector_field,
                embed_rate=embed_rate,
                ids_file=None  # 不使用ID文件
            )

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
