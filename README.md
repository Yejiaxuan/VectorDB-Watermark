# DbWM - 数据库水印管理系统

基于深度学习的数据库水印嵌入与提取系统，支持PostgreSQL向量数据库的水印管理功能。

## 项目结构

```
DbWM/
├── algorithms/                      # 算法模块
│   └── deep_learning/               # 深度学习水印算法
│       ├── dataset.py               # 数据集处理
│       ├── decoder.py               # 水印解码器
│       ├── encoder.py               # 水印编码器
│       ├── noise_layers.py          # 噪声层实现
│       ├── trainer.py               # 模型训练器
│       ├── watermark.py             # 水印核心功能
│       └── results/
│           └── vector_val/
│               └── best.pt          # 预训练模型
├── backend/                         # 后端API服务
│   ├── app.py                       # FastAPI应用主文件
│   └── models.py                    # 数据模型定义
├── database/                        # 数据库模块
│   └── pgvector/                    # PostgreSQL向量数据库
│       ├── client.py                # 数据库客户端
│       ├── pg_func.py               # 水印相关数据库函数
│       └── Docker/
│           ├── docker-compose.yml   # Docker配置
│           └── init.sql             # 数据库初始化脚本
├── frontend/                          # 前端用户界面
│   ├── src/
│   │   ├── pages/
│   │   │   ├── MilvusPage.jsx       # Milvus管理页面
│   │   │   └── PgvectorPage.jsx     # Pgvector管理页面
│   │   ├── App.jsx                  # 主应用组件
│   │   ├── main.jsx                 # 应用入口
│   │   └── api.js                   # API接口封装
│   ├── package.json                 # 前端依赖
│   └── vite.config.js               # Vite配置
├── run.py                           # 项目启动脚本
├── requirements.txt                 # Python依赖
└── environment.yml                  # Conda环境配置
```

## 环境要求

- Python 3.10
- Node.js 16+
- Docker & Docker Compose

## 复现步骤

### 1. 环境配置

#### 使用Conda环境
```bash
# 创建并激活conda环境
conda env create -f environment.yml
conda activate dbwm
```

#### 或使用pip安装
```bash
# 安装Python依赖
pip install -r requirements.txt
```

### 2. 数据库配置

#### 启动PostgreSQL数据库
```bash
cd database/pgvector/Docker
docker-compose up -d
```

等待数据库启动完成后，数据库将在 `localhost:5432` 运行，默认用户名和密码参见 `docker-compose.yml`。

### 3. 前后端服务启动

```bash
# 在项目根目录运行
python run.py
```

后端API将在 `http://localhost:8000` 启动。
前端界面将在 `http://localhost:5173` 启动。


### 4. 访问系统

打开浏览器访问 `http://localhost:5173`，即可使用数据库水印管理系统。
