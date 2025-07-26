import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .models import DBParams, WatermarkEmbedRequest, WatermarkExtractRequest, MilvusDBParams, \
    MilvusWatermarkEmbedRequest, MilvusWatermarkExtractRequest
from database.pgvector.client import PGVectorManager
from database.milvus.client import MilvusManager
# 延迟导入训练模块以加快启动速度
# from algorithms.deep_learning.trainer import train_from_database
from configs.config import Config
import threading

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

# 用于跟踪训练状态的全局变量
training_status = {}
training_lock = threading.Lock()


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
    在指定表的向量列中嵌入水印，使用AES-GCM加密明文消息
    """
    result = pgvector_manager.embed_watermark(
        db_params=request.db_params,
        table=request.table,
        id_column=request.id_column,
        vector_column=request.vector_column,
        message=request.message,
        embed_rate=request.embed_rate,
        encryption_key=request.encryption_key
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


# 移除了下载ID文件的API端点，因为不再需要ID文件


@app.post("/api/extract-watermark")
async def extract_watermark_api(request: WatermarkExtractRequest):
    """
    从指定表的向量列中提取水印，使用AES-GCM解密得到明文消息
    """
    result = pgvector_manager.extract_watermark(
        db_params=request.db_params,
        table=request.table,
        id_column=request.id_column,
        vector_column=request.vector_column,
        embed_rate=request.embed_rate,
        encryption_key=request.encryption_key,
        nonce=request.nonce
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/get_vector_dimension")
async def get_vector_dimension_api(
        params: DBParams,
        table: str = Query(..., description="表名"),
        vector_column: str = Query(..., description="向量列名")
):
    """
    获取PGVector数据库中指定表向量列的维度
    """
    result = pgvector_manager.get_vector_dimension(params.dict(), table, vector_column)
    if result["success"]:
        return {"dimension": result["dimension"]}
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/check_model")
async def check_model_api(dimension: int = Query(..., description="向量维度")):
    """
    检查指定维度的模型是否存在
    """
    try:
        model_path = Config.get_model_path(dimension)
        exists = os.path.exists(model_path)
        return {
            "exists": exists,
            "model_path": model_path,
            "dimension": dimension
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/train_model")
async def train_model_api(
        background_tasks: BackgroundTasks,
        params: DBParams,
        table: str = Query(..., description="表名"),
        vector_column: str = Query(..., description="向量列名"),
        dimension: int = Query(..., description="向量维度"),
        epochs: int = Query(100, description="训练轮数"),
        learning_rate: float = Query(0.0003, description="学习率"),
        batch_size: int = Query(8192, description="批处理大小"),
        val_ratio: float = Query(0.15, description="验证集比例")
):
    """
    使用PGVector数据库中的数据训练模型
    """
    # 生成训练任务ID
    task_id = f"pgvector_{table}_{vector_column}_{dimension}"
    
    with training_lock:
        if task_id in training_status:
            if training_status[task_id]["status"] == "running":
                raise HTTPException(status_code=409, detail="该配置的训练任务正在进行中")
        
        training_status[task_id] = {
            "status": "starting",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": epochs,
            "message": "正在获取训练数据...",
            "dimension": dimension,
            "train_params": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "val_ratio": val_ratio
            },
            "metrics": {
                "train_loss": 0,
                "train_ber": 1,
                "val_loss": 0,
                "val_ber": 1
            }
        }
    
    # 在后台执行训练
    background_tasks.add_task(
        _train_pgvector_model,
        task_id,
        params.dict(),
        table,
        vector_column,
        dimension,
        epochs,
        learning_rate,
        batch_size,
        val_ratio
    )
    
    return {
        "task_id": task_id,
        "message": "训练任务已启动",
        "dimension": dimension,
        "train_params": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "val_ratio": val_ratio
        }
    }


@app.get("/api/training_status/{task_id}")
async def get_training_status(task_id: str):
    """
    获取训练任务状态
    """
    if task_id not in training_status:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    
    return training_status[task_id]


def _train_pgvector_model(task_id: str, db_params: dict, table: str, vector_column: str, 
                         dimension: int, epochs: int, learning_rate: float, 
                         batch_size: int, val_ratio: float):
    """
    训练PGVector模型的后台任务
    """
    try:
        # 延迟导入训练模块以加快启动速度
        from algorithms.deep_learning.trainer import train_from_database
        # 更新状态
        training_status[task_id]["status"] = "running"
        training_status[task_id]["message"] = "正在获取训练数据..."
        
        # 获取训练数据
        result = pgvector_manager.get_training_data(db_params, table, vector_column)
        if not result["success"]:
            training_status[task_id] = {
                "status": "failed",
                "error": f"获取训练数据失败：{result['error']}",
                "dimension": dimension
            }
            return
        
        vectors = result["vectors"]
        training_status[task_id]["message"] = f"获取到 {len(vectors)} 个向量，开始训练..."
        
        # 定义进度回调函数
        def progress_callback(current_epoch, total_epochs, metrics):
            progress = int((current_epoch / total_epochs) * 100)
            training_status[task_id].update({
                "progress": progress,
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "message": f"训练中... Epoch {current_epoch}/{total_epochs}",
                "metrics": metrics
            })
        
        # 开始训练
        train_result = train_from_database(
            vectors, dimension, f"pgvector_{table}",
            epochs=epochs, learning_rate=learning_rate,
            batch_size=batch_size, val_ratio=val_ratio,
            progress_callback=progress_callback
        )
        
        if train_result["success"]:
            training_status[task_id] = {
                "status": "completed",
                "progress": 100,
                "current_epoch": epochs,
                "total_epochs": epochs,
                "message": f"训练完成！最佳BER: {train_result['best_ber']:.3%}",
                "model_path": train_result["model_path"],
                "best_ber": train_result["best_ber"],
                "epochs": train_result["epochs"],
                "dimension": dimension,
                "performance_level": train_result["performance_level"],
                "suggestions": train_result["suggestions"],
                "final_metrics": train_result["final_metrics"],
                "train_params": {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "val_ratio": val_ratio
                }
            }
        else:
            training_status[task_id] = {
                "status": "failed",
                "error": "训练失败",
                "dimension": dimension
            }
    except Exception as e:
        training_status[task_id] = {
            "status": "failed",
            "error": str(e),
            "dimension": dimension
        }


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
    在指定Milvus集合的向量字段中嵌入水印，使用AES-GCM加密明文消息
    """
    result = milvus_manager.embed_watermark(
        db_params=request.db_params,
        collection_name=request.collection_name,
        id_field=request.id_field,
        vector_field=request.vector_field,
        message=request.message,
        embed_rate=request.embed_rate,
        encryption_key=request.encryption_key
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/milvus/extract_watermark")
async def extract_milvus_watermark_api(request: MilvusWatermarkExtractRequest):
    """
    从指定Milvus集合的向量字段中提取水印，使用AES-GCM解密得到明文消息
    """
    result = milvus_manager.extract_watermark(
        db_params=request.db_params,
        collection_name=request.collection_name,
        id_field=request.id_field,
        vector_field=request.vector_field,
        embed_rate=request.embed_rate,
        encryption_key=request.encryption_key,
        nonce=request.nonce
    )

    if result["success"]:
        return result
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/milvus/get_vector_dimension")
async def get_milvus_vector_dimension_api(
        params: MilvusDBParams,
        collection_name: str = Query(..., description="集合名"),
        vector_field: str = Query(..., description="向量字段名")
):
    """
    获取Milvus数据库中指定集合向量字段的维度
    """
    result = milvus_manager.get_vector_dimension(params.dict(), collection_name, vector_field)
    if result["success"]:
        return {"dimension": result["dimension"]}
    else:
        raise HTTPException(status_code=400, detail=result["error"])


@app.post("/api/milvus/train_model")
async def train_milvus_model_api(
        background_tasks: BackgroundTasks,
        params: MilvusDBParams,
        collection_name: str = Query(..., description="集合名"),
        vector_field: str = Query(..., description="向量字段名"),
        dimension: int = Query(..., description="向量维度"),
        epochs: int = Query(100, description="训练轮数"),
        learning_rate: float = Query(0.0003, description="学习率"),
        batch_size: int = Query(8192, description="批处理大小"),
        val_ratio: float = Query(0.15, description="验证集比例")
):
    """
    使用Milvus数据库中的数据训练模型
    """
    # 生成训练任务ID
    task_id = f"milvus_{collection_name}_{vector_field}_{dimension}"
    
    with training_lock:
        if task_id in training_status:
            if training_status[task_id]["status"] == "running":
                raise HTTPException(status_code=409, detail="该配置的训练任务正在进行中")
        
        training_status[task_id] = {
            "status": "starting",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": epochs,
            "message": "正在获取训练数据...",
            "dimension": dimension,
            "train_params": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "val_ratio": val_ratio
            },
            "metrics": {
                "train_loss": 0,
                "train_ber": 1,
                "val_loss": 0,
                "val_ber": 1
            }
        }
    
    # 在后台执行训练
    background_tasks.add_task(
        _train_milvus_model,
        task_id,
        params.dict(),
        collection_name,
        vector_field,
        dimension,
        epochs,
        learning_rate,
        batch_size,
        val_ratio
    )
    
    return {
        "task_id": task_id,
        "message": "训练任务已启动",
        "dimension": dimension,
        "train_params": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "val_ratio": val_ratio
        }
    }


def _train_milvus_model(task_id: str, db_params: dict, collection_name: str, vector_field: str, 
                       dimension: int, epochs: int, learning_rate: float, 
                       batch_size: int, val_ratio: float):
    """
    训练Milvus模型的后台任务
    """
    try:
        # 延迟导入训练模块以加快启动速度
        from algorithms.deep_learning.trainer import train_from_database
        # 更新状态
        training_status[task_id]["status"] = "running"
        training_status[task_id]["message"] = "正在获取训练数据..."
        
        # 获取训练数据
        result = milvus_manager.get_training_data(db_params, collection_name, vector_field)
        if not result["success"]:
            training_status[task_id] = {
                "status": "failed",
                "error": f"获取训练数据失败：{result['error']}",
                "dimension": dimension
            }
            return
        
        vectors = result["vectors"]
        training_status[task_id]["message"] = f"获取到 {len(vectors)} 个向量，开始训练..."
        
        # 定义进度回调函数
        def progress_callback(current_epoch, total_epochs, metrics):
            progress = int((current_epoch / total_epochs) * 100)
            training_status[task_id].update({
                "progress": progress,
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "message": f"训练中... Epoch {current_epoch}/{total_epochs}",
                "metrics": metrics
            })
        
        # 开始训练
        train_result = train_from_database(
            vectors, dimension, f"milvus_{collection_name}",
            epochs=epochs, learning_rate=learning_rate,
            batch_size=batch_size, val_ratio=val_ratio,
            progress_callback=progress_callback
        )
        
        if train_result["success"]:
            training_status[task_id] = {
                "status": "completed",
                "progress": 100,
                "current_epoch": epochs,
                "total_epochs": epochs,
                "message": f"训练完成！最佳BER: {train_result['best_ber']:.3%}",
                "model_path": train_result["model_path"],
                "best_ber": train_result["best_ber"],
                "epochs": train_result["epochs"],
                "dimension": dimension,
                "performance_level": train_result["performance_level"],
                "suggestions": train_result["suggestions"],
                "final_metrics": train_result["final_metrics"],
                "train_params": {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "val_ratio": val_ratio
                }
            }
        else:
            training_status[task_id] = {
                "status": "failed",
                "error": "训练失败",
                "dimension": dimension
            }
    except Exception as e:
        training_status[task_id] = {
            "status": "failed",
            "error": str(e),
            "dimension": dimension
        }
