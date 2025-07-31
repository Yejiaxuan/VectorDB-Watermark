import psycopg2
import numpy as np
import random
import ast
from tqdm import tqdm

# === 数据库连接参数 ===
DB_HOST = "localhost"
DB_PORT = 5555
DB_NAME = "pgvector"
DB_USER = "pgvector"
DB_PASSWORD = "pgvector"
TABLE_NAME = "items"
VECTOR_COLUMN = "emb"
PRIMARY_KEY = "id"  # 改成你自己的主键字段名

# === 替换比例 ===
REPLACE_RATIO = 0.1  # 替换10%的向量

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
print("正在读取全部向量...")
cur.execute(f"SELECT {PRIMARY_KEY}, {VECTOR_COLUMN} FROM {TABLE_NAME}")
data = cur.fetchall()
ids, vectors = zip(*data)
vectors = np.array([
    np.array(ast.literal_eval(v)).astype(float) if isinstance(v, str) else np.array(v).astype(float)
    for v in vectors
])
# === 计算均值向量 ===
print("计算均值向量...")
mean_vector = np.mean(vectors, axis=0)

# === 随机选择部分数据替换 ===
replace_count = int(len(ids) * REPLACE_RATIO)
replace_indices = random.sample(range(len(ids)), replace_count)
replace_ids = [ids[i] for i in replace_indices]

print(f"将替换 {replace_count} 条向量为均值...")

for idx in tqdm(replace_ids):
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {VECTOR_COLUMN} = %s WHERE {PRIMARY_KEY} = %s",
        (list(mean_vector), idx)
    )

conn.commit()
print("✅ 均值替换攻击完成。")
cur.close()
conn.close()
