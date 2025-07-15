from pydantic import BaseModel
from typing import Dict, Any

class DBParams(BaseModel):
    host: str
    port: int
    dbname: str
    user: str
    password: str


# Milvus连接参数模型
class MilvusDBParams(BaseModel):
    host: str
    port: int


# 水印嵌入请求模型
class WatermarkEmbedRequest(BaseModel):
    db_params: Dict[str, Any]  # 数据库连接参数
    table: str  # 表名
    id_column: str  # 主键列名
    vector_column: str  # 向量列名
    message: str  # 水印消息
    total_vecs: int = 1600  # 使用的向量数量，默认1600


# Milvus水印嵌入请求模型
class MilvusWatermarkEmbedRequest(BaseModel):
    db_params: Dict[str, Any]  # Milvus连接参数
    collection_name: str  # 集合名
    id_field: str  # 主键字段名
    vector_field: str  # 向量字段名
    message: str  # 水印消息
    total_vecs: int = 1600  # 使用的向量数量，默认1600
