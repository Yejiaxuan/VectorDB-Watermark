# DbWM

## 项目结构

```
DbWM/
├── algorithms/                    # 算法模块
│   ├── deep_learning/            # 深度学习相关算法
│   │   ├── decoder.py           # 解码器
│   │   ├── encoder.py           # 编码器
│   │   ├── noise_layers.py      # 噪声层
│   │   └── trainer.py           # 训练器
│   └── __init__.py
├── core/                         # 核心模块
│   ├── dataset.py               # 数据集处理
│   ├── watermark.py             # 水印核心功能
│   └── __init__.py
├── database/                     # 数据库模块
│   ├── pgvector/                # PostgreSQL 向量数据库
│   │   ├── insert_npy_vectors.py # 向量插入脚本
│   │   ├── insert_script.py     # 数据插入脚本
│   │   └── tests/               # 测试文件
│   │       ├── in_degree_test.py
│   │       ├── text_test.py
│   │       └── watermark_test.py
│   └── __init__.py
├── web_ui/                       # Web 用户界面
│   ├── src/                     # 源代码
│   │   ├── pages/               # 页面组件
│   │   │   ├── MilvusPage.jsx   # Milvus 数据库页面
│   │   │   └── PgvectorPage.jsx # Pgvector 数据库页面
│   │   ├── App.jsx              # 主应用组件
│   │   ├── main.jsx             # 入口文件
│   │   └── api.js               # API 接口
│   ├── public/                  # 静态资源
│   ├── package.json             # 项目依赖配置
│   └── vite.config.js           # Vite 构建配置
└── README.md                     # 项目说明文档
```
