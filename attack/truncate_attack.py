import psycopg2
import numpy as np
import ast
from tqdm import tqdm
import random

# === 数据库连接参数 ===
DB_HOST = "localhost"
DB_PORT = 5555
DB_NAME = "pgvector"
DB_USER = "pgvector"
DB_PASSWORD = "pgvector"
TABLE_NAME = "items"
VECTOR_COLUMN = "emb"
PRIMARY_KEY = "id"

# === 攻击参数 ===
ORIGINAL_DIM = 384
TRUNCATED_DIM = 284  # 保留前 284 维
PADDING_METHOD = "zero"  # 可选: "zero", "mean"

# === 连接数据库 ===
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cur = conn.cursor()

# === 获取全部向量和主键 ===
print("📥 正在读取向量...")
cur.execute(f"SELECT {PRIMARY_KEY}, {VECTOR_COLUMN} FROM {TABLE_NAME}")
data = cur.fetchall()

ids, vectors = zip(*data)
vectors = np.array([
    np.array(ast.literal_eval(v)).astype(float) if isinstance(v, str) else np.array(v).astype(float)
    for v in vectors
])

assert vectors.shape[1] == ORIGINAL_DIM, f"向量维度错误，应为 {ORIGINAL_DIM}"

# === 计算均值向量（备用补齐方式） ===
mean_vector = np.mean(vectors, axis=0)

# === 执行维度截断攻击 ===
print(f"✂️ 正在对 {len(vectors)} 条向量执行维度截断攻击（保留 {TRUNCATED_DIM} 维）...")
for idx, vec_id in tqdm(enumerate(ids), total=len(ids)):
    truncated = vectors[idx][:TRUNCATED_DIM]

    if PADDING_METHOD == "zero":
        padded = np.concatenate([truncated, np.zeros(ORIGINAL_DIM - TRUNCATED_DIM)])
    elif PADDING_METHOD == "mean":
        padded = np.concatenate([truncated, mean_vector[TRUNCATED_DIM:]])
    else:
        raise ValueError("未知的 PADDING_METHOD，请设置为 'zero' 或 'mean'")

    # 写回数据库
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {VECTOR_COLUMN} = %s WHERE {PRIMARY_KEY} = %s",
        (padded.tolist(), vec_id)
    )

conn.commit()
print("✅ 维度截断攻击完成。")
cur.close()
conn.close()
