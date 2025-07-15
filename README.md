# DbWM - 数据库水印管理系统

基于深度学习的数据库水印嵌入与提取系统，支持PostgreSQL (PGVector) 和 Milvus 向量数据库的水印管理功能。

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
│       ├── watermark_embedding.py   # 水印嵌入功能
│       ├── test.py                  # 性能测试脚本
│       └── results/                 # 训练结果目录
│           └── vector_val/          # 验证结果
│               └── best.pt          # 预训练模型
├── backend/                         # 后端API服务
│   ├── app.py                       # FastAPI应用主文件
│   └── models.py                    # 数据模型定义
├── database/                        # 数据库模块
│   ├── pgvector/                    # PostgreSQL向量数据库
│   │   ├── client.py                # 数据库客户端
│   │   ├── pg_func.py               # 水印相关数据库函数
│   │   └── docker/
│   │       ├── docker-compose.yml   # Docker配置
│   │       └── init.sql             # 数据库初始化脚本
│   └── milvus/                      # Milvus向量数据库
│       ├── client.py                # Milvus客户端管理器
│       ├── milvus_func.py           # Milvus水印核心功能
│       └── docker/
│           ├── milvus-standalone-docker-compose-gpu.yml  # Milvus Docker配置
│           └── nq_qa_combined_384d.npy                   # 测试数据文件
├── frontend/                        # 前端用户界面
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

#### 方式A：使用PostgreSQL (PGVector)

```bash
cd database/pgvector/docker
docker-compose -p pgvector up -d
```

等待数据库启动完成后，数据库将在 `localhost:5432` 运行，默认用户名和密码参见 `docker-compose.yml`。

#### 方式B：使用Milvus

```bash
cd database/milvus/docker
docker-compose -f milvus-standalone-docker-compose-gpu.yml up -d
```

等待Milvus服务启动完成后，服务将在 `localhost:19530` 运行。

### 3. 数据导入 (测试Milvus需要)

如果选择使用Milvus，需要导入测试数据：

```bash
python database/milvus/insert_nq_data.py
```

该脚本会：
- 连接到Milvus (localhost:19530)
- 创建名为 `nq_qa_combined` 的集合
- 导入100,231个384维向量
- 创建向量索引

### 4. 前后端服务启动

```bash
# 在项目根目录运行
python run.py
```

后端API将在 `http://localhost:8000` 启动。
前端界面将在 `http://localhost:5173` 启动。

### 5. 访问系统

打开浏览器访问 `http://localhost:5173`，你会看到两个选项：
- **PGVector 数据库** (青色按钮)
- **Milvus 数据库** (紫色按钮)

## 使用指南

### PGVector 数据库使用

1. 点击"PGVector 数据库"按钮
2. 配置数据库连接参数
3. 进行水印嵌入和提取操作

### Milvus 数据库使用

#### 步骤1：连接数据库
- 主机地址：`localhost` (默认)
- 端口：`19530` (默认)
- 点击"连接数据库"

#### 步骤2：配置数据源
- **选择集合**：`nq_qa_combined` (如果使用提供的数据)
- **选择主键字段**：`id`
- **选择向量字段**：`embedding`

#### 步骤3：水印操作

##### 嵌入水印
1. 输入32个字符的水印消息
2. 点击"嵌入水印并下载ID文件"
3. 系统会自动下载ID文件，请妥善保存

##### 提取水印
1. 点击文件上传区域，选择之前下载的ID文件
2. 点击"提取水印"
3. 查看提取结果

## 技术架构

### 系统架构

```
frontend/src/pages/MilvusPage.jsx     # Milvus前端页面
frontend/src/pages/PgvectorPage.jsx   # PGVector前端页面
database/milvus/client.py             # Milvus客户端管理器
database/milvus/milvus_func.py        # Milvus水印核心功能
database/pgvector/client.py           # PGVector客户端
database/pgvector/pg_func.py          # PGVector水印功能
backend/app.py                         # 后端API (支持两种数据库)
frontend/src/api.js                    # 前端API调用
```

### 数据结构

#### Milvus数据结构
- **集合名称**：`nq_qa_combined`
- **向量维度**：384
- **数据量**：100,231条记录
- **字段**：
  - `id` (INT64, 主键)
  - `embedding` (FLOAT_VECTOR, 384维)

#### 水印算法
- 使用统一的深度学习水印算法
- 基于向量图的入度分析
- 支持32字符水印消息
- 16个数据块，每块16位载荷

### API端点

#### Milvus API
- `POST /api/milvus/connect` - 连接测试
- `POST /api/milvus/collections` - 获取集合列表
- `POST /api/milvus/vector_fields` - 获取向量字段
- `POST /api/milvus/primary_keys` - 获取主键字段
- `POST /api/milvus/embed_watermark` - 嵌入水印
- `POST /api/milvus/extract_watermark_with_file` - 提取水印

#### PGVector API
- `POST /api/pgvector/connect` - 连接测试
- `POST /api/pgvector/tables` - 获取表列表
- `POST /api/pgvector/vector_columns` - 获取向量列
- `POST /api/pgvector/embed_watermark` - 嵌入水印
- `POST /api/pgvector/extract_watermark` - 提取水印


## 测试结果

### 深度学习水印算法性能测试

```bash
# 在项目根目录运行
python -m algorithms.deep_learning.test
```

#### 1. 嵌入质量（余弦相似度）
- **平均相似度**: 0.970701
- **最小相似度**: 0.964209
- **最大相似度**: 0.978496
- **标准差**: 0.002335

#### 2. 水印提取准确率
- **比特错误率**: 0.000000
- **容错阈值 1 比特的准确率**: 100.00%

#### 3. CRC校验成功率
- **成功率**: 100.00%

#### 4. 噪声鲁棒性测试

| 噪声类型 | 比特错误率 | 容错阈值 1 比特的准确率 | CRC校验成功率 |
|----------|------------|------------------------|---------------|
| 无噪声 | 0.000042 | 100.00% | 100.00% |
| 高斯噪声(0.01) | 0.000000 | 100.00% | 100.00% |
| 高斯噪声(0.02) | 0.000292 | 100.00% | 100.00% |
| 量化(12) | 0.000000 | 100.00% | 100.00% |
| 量化(10) | 0.000042 | 100.00% | 100.00% |
| 量化(8) | 0.000000 | 100.00% | 100.00% |
| 维度遮蔽(0.95) | 0.001417 | 99.90% | 99.30% |
| 维度遮蔽(0.90) | 0.007917 | 98.10% | 97.00% |
| 组合噪声1 | 0.000000 | 100.00% | 100.00% |
| 组合噪声2 | 0.003542 | 98.80% | 98.70% |
| 组合噪声3 | 0.009375 | 97.30% | 96.00% |

#### 5. 时间性能

- **单样本嵌入时间**: 1.499 毫秒
- **单样本提取时间**: 0.660 毫秒
- **批处理嵌入时间（每样本）**: 0.024 毫秒
- **批处理提取时间（每样本）**: 0.011 毫秒

### 测试结论

1. **高质量嵌入**: 余弦相似度均值达到 0.97，表明水印嵌入对原始向量质量影响极小
2. **零错误提取**: 在无噪声条件下实现完美的水印提取，比特错误率为 0
3. **强噪声鲁棒性**: 在各种噪声条件下均保持较高的准确率，即使在最强的组合噪声3下仍能达到 96% 的CRC校验成功率
4. **高效处理**: 批处理模式下每样本嵌入仅需 0.024 毫秒，提取仅需 0.011 毫秒，满足实时应用需求
5. **多数据库支持**: 系统同时支持PGVector和Milvus，为不同场景提供灵活选择

## 支持

如遇问题，请检查：
1. 所有服务是否正常启动
2. 依赖包是否正确安装
3. 网络连接是否正常
4. 数据文件是否完整

---

🎉 **恭喜！你现在可以在PGVector和Milvus数据库中使用向量水印技术了！**
