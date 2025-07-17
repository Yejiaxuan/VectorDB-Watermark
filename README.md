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

## 使用说明

1. 打开 http://localhost:5173
2. 选择数据库类型（PGVector 或 Milvus）
3. 连接数据库
4. 进行水印嵌入/提取操作

### Milvus 使用提示
- 集合选择：`nq_qa_combined`
- 主键字段：`id`
- 向量字段：`embedding`
- 水印消息：32个字符

### PGVector 使用提示
- 确保数据库已启动（端口5432）
- 配置连接参数后即可使用

## 项目结构

```
DbWM/
├── algorithms/deep_learning/    # 深度学习水印算法
├── backend/                     # FastAPI后端
├── frontend/                    # React前端
├── database/                    # 数据库模块
│   ├── pgvector/               # PostgreSQL支持
│   └── milvus/                 # Milvus支持
└── run.py                      # 启动脚本
```

## 常见问题

1. **数据库连接失败**：检查Docker容器是否启动
2. **Milvus集合为空**：运行 `insert_nq_data.py` 导入数据
3. **前端无法访问**：确认端口5173未被占用
4. **依赖安装失败**：使用conda环境或检查Python版本(需要3.10)

---
有问题直接找我 🚀
