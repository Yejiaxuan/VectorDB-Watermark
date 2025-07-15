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

    def __init__(self, temp_dir: str = "temp_files"):
        """
        初始化管理器
        
        Args:
            temp_dir: 临时文件目录
        """
        self.temp_dir = temp_dir
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
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

    def embed_watermark_with_file(
            self,
            db_params: Dict[str, Any],
            collection_name: str,
            id_field: str,
            vector_field: str,
            message: str,
            total_vecs: int = 1600
    ) -> Dict[str, Any]:
        """
        在指定集合的向量字段中嵌入水印，并生成唯一ID文件
        
        Args:
            db_params: 数据库连接参数
            collection_name: 集合名
            id_field: 主键字段名
            vector_field: 向量字段名
            message: 水印消息
            total_vecs: 使用的向量数量，默认1600
            
        Returns:
            嵌入结果字典
        """
        try:
            # 生成带会话ID的唯一文件名
            session_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            ids_file = f"{self.temp_dir}/wm_{collection_name}_{vector_field}_{timestamp}_{session_id}.json"

            # 调用水印嵌入函数
            result = embed_watermark(
                db_params=db_params,
                collection_name=collection_name,
                id_field=id_field,
                vector_field=vector_field,
                message=message,
                total_vecs=total_vecs,
                ids_file=ids_file
            )

            # 检查结果
            if not result["success"]:
                # 如果嵌入失败，尝试删除可能创建的文件
                if os.path.exists(ids_file):
                    os.remove(ids_file)
                return result

            # 设置文件名以便于客户端下载
            result["file_id"] = os.path.basename(ids_file)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_watermark_with_uploaded_file(
            self,
            db_params: Dict[str, Any],
            collection_name: str,
            id_field: str,
            vector_field: str,
            file_content: bytes
    ) -> Dict[str, Any]:
        """
        从指定集合的向量字段中提取水印，使用上传的ID文件
        
        Args:
            db_params: 数据库连接参数
            collection_name: 集合名
            id_field: 主键字段名
            vector_field: 向量字段名
            file_content: 上传的文件内容
            
        Returns:
            提取结果字典
        """
        temp_file_path = None
        try:
            # 生成唯一的临时文件名
            session_id = str(uuid.uuid4())
            temp_file_path = f"{self.temp_dir}/temp_{session_id}.json"

            # 保存上传的文件到临时位置
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

            # 调用水印提取函数
            result = extract_watermark(
                db_params=db_params,
                collection_name=collection_name,
                id_field=id_field,
                vector_field=vector_field,
                ids_file=temp_file_path
            )

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            # 无论成功与否，确保清理临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass  # 如果删除失败，不要中断响应

    def get_file_path(self, file_id: str) -> str:
        """
        获取文件的完整路径
        
        Args:
            file_id: 文件ID
            
        Returns:
            文件的完整路径
        """
        return f"{self.temp_dir}/{file_id}"

    def file_exists(self, file_id: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            file_id: 文件ID
            
        Returns:
            文件是否存在
        """
        return os.path.exists(self.get_file_path(file_id)) 