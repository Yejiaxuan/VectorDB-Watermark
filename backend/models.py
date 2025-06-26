from pydantic import BaseModel
from typing import Dict, Any

class DBParams(BaseModel):
    host: str
    port: int
    dbname: str
    user: str
    password: str


# 水印嵌入请求模型
class WatermarkEmbedRequest(BaseModel):
    db_params: Dict[str, Any]  # 数据库连接参数
    table: str  # 表名
    id_column: str  # 主键列名
    vector_column: str  # 向量列名
    message: str  # 水印消息
    total_vecs: int = 1600  # 使用的向量数量，默认1600
