#!/usr/bin/env python3
"""
insert_npy_vectors.py

从指定的 .npy 文件中加载已生成的 384 维向量，并批量插入到 pgvector 的 items 表（emb 列）。
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
TABLE_NAME   = os.getenv('TABLE_NAME', 'items')
VECTOR_COLUMN = os.getenv('VECTOR_COLUMN', 'emb')
NPY_PATH     = os.getenv('NPY_PATH', "HiDDeN/nq_qa_combined_384d.npy")
# ————————

def main():
    # 加载 .npy 文件
    if not os.path.isfile(NPY_PATH):
        raise FileNotFoundError(f"向量文件未找到: {NPY_PATH}")
    data = np.load(NPY_PATH)
    # 确保维度正确
    if data.ndim != 2 or data.shape[1] != 384:
        raise ValueError(f"期望形状 (N,384)，但得到 {data.shape}")

    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)  # 注册 VECTOR 类型
    cur = conn.cursor()

    # 将 numpy 数组转换为要插入的值列表
    values = [(vec.tolist(),) for vec in data.astype(np.float32)]

    # 批量插入
    sql = f"INSERT INTO {TABLE_NAME} ({VECTOR_COLUMN}) VALUES %s;"
    execute_values(cur, sql, values, template="(%s)")
    conn.commit()

    print(f"成功插入 {len(data)} 条向量到表 `{TABLE_NAME}`（列 `{VECTOR_COLUMN}`）。")

    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
