from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
from .models import DBParams
from database.pgvector.client import PGVectorManager  # 导入PGVector管理器

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建管理器实例
database_manager = PGVectorManager(temp_dir="temp_files")


@app.post("/api/connect")
async def connect_db(params: DBParams):
    result = database_manager.test_connection(params.dict())
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/tables")
async def list_tables(params: DBParams):
    """
    列出 public schema 下所有表名
    """
    result = database_manager.list_tables(params.dict())
    if result["success"]:
        return {"tables": result["tables"]}
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/columns")
async def list_columns(
        params: DBParams,
        table: str = Query(..., description="要查询的表名")
):
    """
    列出指定表中所有 pgvector 类型的列
    """
    result = database_manager.list_vector_columns(params.dict(), table)
    if result["success"]:
        return {"columns": result["columns"]}
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/primarykeys")
async def list_primary_keys(
        params: DBParams,
        table: str = Query(..., description="要查询的表名")
):
    """
    列出指定表的主键列
    """
    result = database_manager.list_primary_keys(params.dict(), table)
    if result["success"]:
        return {"keys": result["keys"]}
    else:
        raise HTTPException(status_code=400, detail=result["error"])


# 水印嵌入请求模型
class WatermarkEmbedRequest(BaseModel):
    db_params: Dict[str, Any]  # 数据库连接参数
    table: str  # 表名
    id_column: str  # 主键列名
    vector_column: str  # 向量列名
    message: str  # 水印消息
    total_vecs: int = 1600  # 使用的向量数量，默认1600


@app.post("/api/embed_watermark")
async def embed_watermark_api(request: WatermarkEmbedRequest):
    """
    在指定表的向量列中嵌入水印，并生成唯一ID文件
    """
    result = database_manager.embed_watermark_with_file(
        db_params=request.db_params,
        table=request.table,
        id_column=request.id_column,
        vector_column=request.vector_column,
        message=request.message,
        total_vecs=request.total_vecs
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.get("/api/download_ids_file/{file_id}")
async def download_ids_file(file_id: str):
    """
    下载水印ID文件（使用文件ID）
    """
    if not database_manager.file_exists(file_id):
        raise HTTPException(status_code=404, detail=f"ID文件不存在或已过期")

    file_path = database_manager.get_file_path(file_id)
    return FileResponse(
        path=file_path,
        filename=file_id,
        media_type="application/json"
    )


@app.post("/api/extract_watermark_with_file")
async def extract_watermark_with_file(
        file: UploadFile = File(...),
        db_json: str = Form(...),
        table: str = Form(...),
        id_column: str = Form(...),
        vector_column: str = Form(...)
):
    """
    从指定表的向量列中提取水印，使用上传的ID文件
    """
    # 解析数据库连接参数
    db_params = json.loads(db_json)

    # 读取上传的文件内容
    file_content = await file.read()

    # 调用管理器的提取方法
    result = database_manager.extract_watermark_with_uploaded_file(
        db_params=db_params,
        table=table,
        id_column=id_column,
        vector_column=vector_column,
        file_content=file_content
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])
