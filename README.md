# DbWM - 数据库水印管理系统

## 项目结构

```
DbWM/
├── algorithms/                    # 算法模块
│   ├── deep_learning/            # 深度学习相关算法
│   │   ├── decoder.py           # 水印解码器
│   │   ├── encoder.py           # 水印编码器
│   │   ├── noise_layers.py      # 噪声层实现
│   │   └── trainer.py           # 模型训练器
│   └── __init__.py
├── backend/                      # 后端服务
│   ├── app.py                   # FastAPI应用入口
│   ├── db_service.py            # 数据库服务层
│   ├── models.py                # 数据模型
│   └── requirements.txt         # 后端依赖
├── core/                         # 核心模块
│   ├── dataset.py               # 数据集处理
│   ├── watermark.py             # 水印核心功能
│   └── __init__.py
├── database/                     # 数据库模块
│   └── pgvector/                # PostgreSQL向量数据库
│       └── pgvector_client.py   # 向量数据库客户端
└── web_ui/                       # Web用户界面
    ├── src/                     # 前端源代码
    │   ├── pages/               # 页面组件
    │   │   ├── MilvusPage.jsx   # Milvus管理页面
    │   │   └── PgvectorPage.jsx # Pgvector管理页面
    │   ├── App.jsx              # 主应用组件
    │   ├── main.jsx             # 入口文件
    │   └── api.js               # API接口
    ├── package.json             # 前端依赖配置
    └── vite.config.js           # Vite构建配置
```