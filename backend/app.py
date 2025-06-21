from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import psycopg2
import os
import json
import uuid
from datetime import datetime
from pgvector.psycopg2 import register_vector
from .models import DBParams
from .pg_func import embed_watermark, extract_watermark  # 导入提取水印函数

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建临时文件目录
TEMP_DIR = "temp_files"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.post("/api/connect")
async def connect_db(params: DBParams):
    try:
        conn = psycopg2.connect(**params.dict())
        register_vector(conn)
        conn.close()
        return {"success": True, "message": "连接成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/tables")
async def list_tables(params: DBParams):
    """
    列出 public schema 下所有表名
    """
    try:
        conn = psycopg2.connect(**params.dict())
        cur = conn.cursor()
        cur.execute(
            """
            SELECT table_name
              FROM information_schema.tables
             WHERE table_schema = 'public'
               AND table_type = 'BASE TABLE';
            """
        )
        tables = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()
        return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/columns")
async def list_columns(
    params: DBParams,
    table: str = Query(..., description="要查询的表名")
):
    """
    列出指定表中所有 pgvector 类型的列
    """
    try:
        conn = psycopg2.connect(**params.dict())
        cur = conn.cursor()
        cur.execute(
            """
            SELECT column_name
              FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = %s
               AND udt_name = 'vector';
            """,
            (table,)
        )
        cols = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()
        return {"columns": cols}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/primarykeys")
async def list_primary_keys(
    params: DBParams,
    table: str = Query(..., description="要查询的表名")
):
    """
    列出指定表的主键列
    """
    try:
        conn = psycopg2.connect(**params.dict())
        cur = conn.cursor()
        cur.execute(
            """
            SELECT kc.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kc
                ON kc.constraint_name = tc.constraint_name
            WHERE
                tc.constraint_type = 'PRIMARY KEY' AND
                tc.table_schema = 'public' AND
                tc.table_name = %s
            ORDER BY kc.ordinal_position;
            """,
            (table,)
        )
        keys = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()
        return {"keys": keys}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 水印嵌入请求模型
class WatermarkEmbedRequest(BaseModel):
    db_params: Dict[str, Any]  # 数据库连接参数
    table: str                 # 表名
    id_column: str             # 主键列名
    vector_column: str         # 向量列名
    message: str               # 水印消息
    total_vecs: int = 1600     # 使用的向量数量，默认1600


@app.post("/api/embed_watermark")
async def embed_watermark_api(request: WatermarkEmbedRequest):
    """
    在指定表的向量列中嵌入水印，并生成唯一ID文件
    """
    try:
        # 生成带会话ID的唯一文件名
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        ids_file = f"{TEMP_DIR}/wm_{request.table}_{request.vector_column}_{timestamp}_{session_id}.json"
        
        # 调用水印嵌入函数
        result = embed_watermark(
            db_params=request.db_params,
            table_name=request.table,
            id_col=request.id_column,
            emb_col=request.vector_column,
            message=request.message,
            total_vecs=request.total_vecs,
            ids_file=ids_file
        )
        
        # 检查结果
        if not result["success"]:
            # 如果嵌入失败，尝试删除可能创建的文件
            if os.path.exists(ids_file):
                os.remove(ids_file)
            raise HTTPException(status_code=400, detail=result["error"])
            
        # 设置文件名以便于客户端下载
        result["file_id"] = os.path.basename(ids_file)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/download_ids_file/{file_id}")
async def download_ids_file(file_id: str):
    """
    下载水印ID文件（使用文件ID）
    """
    try:
        # 构建文件路径
        file_path = f"{TEMP_DIR}/{file_id}"
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"ID文件不存在或已过期")
        
        # 返回文件供下载
        return FileResponse(
            path=file_path, 
            filename=file_id,
            media_type="application/json"
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=str(e))


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
    temp_file_path = None
    try:
        # 解析数据库连接参数
        db_params = json.loads(db_json)
        
        # 生成唯一的临时文件名
        session_id = str(uuid.uuid4())
        temp_file_path = f"{TEMP_DIR}/temp_{session_id}.json"
        
        # 保存上传的文件到临时位置
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 调用水印提取函数
        result = extract_watermark(
            db_params=db_params,
            table_name=table,
            id_col=id_column,
            emb_col=vector_column,
            ids_file=temp_file_path
        )
        
        # 检查结果
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # 无论成功与否，确保清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass  # 如果删除失败，不要中断响应