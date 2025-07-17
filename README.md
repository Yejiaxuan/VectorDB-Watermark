# DbWM - 队内复现指南

基于深度学习的数据库水印系统，支持 PGVector 和 Milvus 两种向量数据库。

## 快速启动

### 1. 环境部署

```bash
pip install -r requirements.txt
```

### 2. 数据库启动


#### PGVector
```bash
cd database/pgvector/docker
docker-compose up -d

# 导入测试数据
python ../insert.py
```

#### Milvus (需要GPU)
```bash
cd database/milvus/docker
docker-compose -f milvus-standalone-docker-compose-gpu.yml up -d

# 导入测试数据
python ../insert.py
```

**可自行修改insert.py为其他数据集，不一定为nq。**

### 3. 启动项目

```bash
# 回到项目根目录
python run.py
```

系统会自动启动：
- 后端API: http://localhost:8000
- 前端界面: http://localhost:5173


在浏览器打开前端页面即可：http://localhost:5173
