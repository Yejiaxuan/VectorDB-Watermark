import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .models import DBParams, WatermarkEmbedRequest, WatermarkExtractRequest, MilvusDBParams, \
    MilvusWatermarkEmbedRequest, MilvusWatermarkExtractRequest
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
pgvector_manager = PGVectorManager()
milvus_manager = MilvusManager()


@app.post("/api/connect")
async def connect_db(params: DBParams):
    result = pgvector_manager.test_connection(params.dict())
    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/tables")
async def list_tables(params: DBParams):
    """
    列出 public schema 下所有表名
    """
    result = pgvector_manager.list_tables(params.dict())
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
    result = pgvector_manager.list_vector_columns(params.dict(), table)
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
    result = pgvector_manager.list_primary_keys(params.dict(), table)
    if result["success"]:
        return {"keys": result["keys"]}
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/embed_watermark")
async def embed_watermark_api(request: WatermarkEmbedRequest):
    """
    在指定表的向量列中嵌入水印，不生成ID文件
    """
    result = pgvector_manager.embed_watermark(
        db_params=request.db_params,
        table=request.table,
        id_column=request.id_column,
        vector_column=request.vector_column,
        message=request.message,
        embed_rate=request.embed_rate
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


# 移除了下载ID文件的API端点，因为不再需要ID文件


@app.post("/api/extract-watermark")
async def extract_watermark_api(request: WatermarkExtractRequest):
    """
    从指定表的向量列中提取水印，重新计算低入度节点
    """
    result = pgvector_manager.extract_watermark(
        db_params=request.db_params,
        table=request.table,
        id_column=request.id_column,
        vector_column=request.vector_column,
        embed_rate=request.embed_rate
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
    在指定Milvus集合的向量字段中嵌入水印，不生成ID文件
    """
    result = milvus_manager.embed_watermark(
        db_params=request.db_params,
        collection_name=request.collection_name,
        id_field=request.id_field,
        vector_field=request.vector_field,
        message=request.message,
        embed_rate=request.embed_rate
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/milvus/extract_watermark")
async def extract_milvus_watermark_api(request: MilvusWatermarkExtractRequest):
    """
    从指定Milvus集合的向量字段中提取水印，重新计算低入度节点
    """
    result = milvus_manager.extract_watermark(
        db_params=request.db_params,
        collection_name=request.collection_name,
        id_field=request.id_field,
        vector_field=request.vector_field,
        embed_rate=request.embed_rate
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])
