import psycopg2
import numpy as np
import random
import ast
from tqdm import tqdm

# === 配置参数 ===
DB_HOST = "localhost"
DB_PORT = 5555
DB_NAME = "pgvector"
DB_USER = "pgvector"
DB_PASSWORD = "pgvector"
TABLE_NAME = "items"
VECTOR_COLUMN = "emb"
PRIMARY_KEY = "id"

REPLACE_RATIO = 0.3        # 替换 10% 的向量
GAUSSIAN_STD = 0.1        # 噪声标准差（越大攻击越强）
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# === 数据库连接 ===
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cur = conn.cursor()

# === 获取向量数据 ===
print("📥 正在读取数据库向量...")
cur.execute(f"SELECT {PRIMARY_KEY}, {VECTOR_COLUMN} FROM {TABLE_NAME}")
data = cur.fetchall()
ids, vectors = zip(*data)
vectors = np.array([
    np.array(ast.literal_eval(v)).astype(float) if isinstance(v, str) else np.array(v).astype(float)
    for v in vectors
])
print(f"✅ 共读取 {len(vectors)} 条向量，维度为 {vectors.shape[1]}")

# === 添加高斯噪声 ===
replace_count = int(len(ids) * REPLACE_RATIO)
replace_indices = random.sample(range(len(ids)), replace_count)
print(f"🎯 将向 {replace_count} 条向量添加高斯噪声（σ={GAUSSIAN_STD}）")

for i in tqdm(replace_indices):
    noise = np.random.normal(loc=0.0, scale=GAUSSIAN_STD, size=vectors[i].shape)
    noisy_vector = vectors[i] + noise
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {VECTOR_COLUMN} = %s WHERE {PRIMARY_KEY} = %s",
        (noisy_vector.tolist(), ids[i])
    )

conn.commit()
print("✅ 高斯噪声攻击完成。")

# === 关闭连接 ===
cur.close()
conn.close()
