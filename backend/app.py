import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import json
from .models import DBParams, WatermarkEmbedRequest, MilvusDBParams, MilvusWatermarkEmbedRequest
from database.pgvector.client import PGVectorManager
from database.milvus.client import MilvusManager

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建管理器实例
database_manager = PGVectorManager(temp_dir="temp_files")
milvus_manager = MilvusManager(temp_dir="temp_files")


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


# ===== Milvus API 端点 =====

@app.post("/api/milvus/connect")
async def connect_milvus_db(params: MilvusDBParams):
    """
    连接Milvus数据库
    """
    result = milvus_manager.test_connection(params.dict())
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/milvus/collections")
async def list_milvus_collections(params: MilvusDBParams):
    """
    列出所有Milvus集合
    """
    result = milvus_manager.list_collections(params.dict())
    if result["success"]:
        return {"collections": result["collections"]}
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/milvus/vector_fields")
async def list_milvus_vector_fields(
        params: MilvusDBParams,
        collection_name: str = Query(..., description="要查询的集合名")
):
    """
    列出指定集合中所有向量类型的字段
    """
    result = milvus_manager.list_vector_fields(params.dict(), collection_name)
    if result["success"]:
        return {"fields": result["fields"]}
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/milvus/primary_keys")
async def list_milvus_primary_keys(
        params: MilvusDBParams,
        collection_name: str = Query(..., description="要查询的集合名")
):
    """
    列出指定集合的主键字段
    """
    result = milvus_manager.list_primary_keys(params.dict(), collection_name)
    if result["success"]:
        return {"keys": result["keys"]}
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/milvus/embed_watermark")
async def embed_milvus_watermark_api(request: MilvusWatermarkEmbedRequest):
    """
    在指定Milvus集合的向量字段中嵌入水印，并生成唯一ID文件
    """
    result = milvus_manager.embed_watermark_with_file(
        db_params=request.db_params,
        collection_name=request.collection_name,
        id_field=request.id_field,
        vector_field=request.vector_field,
        message=request.message,
        total_vecs=request.total_vecs
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.get("/api/milvus/download_ids_file/{file_id}")
async def download_milvus_ids_file(file_id: str):
    """
    下载Milvus水印ID文件（使用文件ID）
    """
    if not milvus_manager.file_exists(file_id):
        raise HTTPException(status_code=404, detail=f"ID文件不存在或已过期")

    file_path = milvus_manager.get_file_path(file_id)
    return FileResponse(
        path=file_path,
        filename=file_id,
        media_type="application/json"
    )


@app.post("/api/milvus/extract_watermark_with_file")
async def extract_milvus_watermark_with_file(
        file: UploadFile = File(...),
        db_json: str = Form(...),
        collection_name: str = Form(...),
        id_field: str = Form(...),
        vector_field: str = Form(...)
):
    """
    从指定Milvus集合的向量字段中提取水印，使用上传的ID文件
    """
    # 解析数据库连接参数
    db_params = json.loads(db_json)

    # 读取上传的文件内容
    file_content = await file.read()

    # 调用管理器的提取方法
    result = milvus_manager.extract_watermark_with_uploaded_file(
        db_params=db_params,
        collection_name=collection_name,
        id_field=id_field,
        vector_field=vector_field,
        file_content=file_content
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])
