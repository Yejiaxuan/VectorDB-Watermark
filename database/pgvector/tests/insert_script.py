#!/usr/bin/env python3
"""
insert_random_384_vectors.py

生成 5000 条随机 384 维向量，并插入到 pgvector 中已创建的 items 表（emb 列）。
"""
import os
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
import numpy as np

# —— 配置区 ——
DB_PARAMS = {
    'host':     os.getenv('PG_HOST', 'localhost'),
    'port':     int(os.getenv('PG_PORT', 5432)),
    'dbname':   os.getenv('PG_DATABASE', 'test'),
    'user':     os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', 'ysj'),
}
TABLE_NAME = 'items'
VECTOR_COLUMN = 'emb'
VECTOR_DIM    = 384
NUM_VECTORS   = 5000
# ————————

def main():
    # 连接数据库
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)         # 注册 pgvector 适配器
    cur = conn.cursor()

    # 生成随机向量
    data = np.random.random((NUM_VECTORS, VECTOR_DIM)).astype(np.float32)

    # 构造待插入数据 [(vec1,), (vec2,), ...]
    values = [(vec.tolist(),) for vec in data]

    # 批量插入
    sql = f"INSERT INTO {TABLE_NAME} ({VECTOR_COLUMN}) VALUES %s;"
    execute_values(cur, sql, values, template="(%s)")

    conn.commit()
    print(f"✅ 插入 {NUM_VECTORS} 条 {VECTOR_DIM} 维随机向量到表 `{TABLE_NAME}` 完成。")

    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
