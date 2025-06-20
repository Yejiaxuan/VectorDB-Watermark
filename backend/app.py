from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import DBParams
from .db_service import test_connect

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

@app.post("/api/connect")
async def connect_db(params: DBParams):
    try:
        test_connect(params)
        return {"success": True, "message": "连接成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
