#!/usr/bin/env python3
"""
find_in_degree.py

从 pgvector 拉取向量，基于 Faiss 的 IndexHNSWFlat 构建 HNSW 图，
统计 Base 层中每个节点被其他节点当作邻居的次数（入度），
并按入度升序输出最小的若干节点。
"""
import os
import psycopg2
import numpy as np
import faiss
from pgvector.psycopg2 import register_vector

# —— 配置区 ——
DB_PARAMS = {
    'host':     os.getenv('PG_HOST',     'localhost'),
    'port':     int(os.getenv('PG_PORT', 5432)),
    'dbname':   os.getenv('PG_DATABASE','test'),
    'user':     os.getenv('PG_USER',    'postgres'),
    'password': os.getenv('PG_PASSWORD','ysj'),
}
TABLE_NAME    = 'items'
VECTOR_COLUMN = 'emb'
DIM           = 384
M             = int(os.getenv('HNSW_M',             16))
EF_CONSTRUCTION = int(os.getenv('HNSW_EF_CONSTRUCTION',200))
EF_SEARCH       = int(os.getenv('HNSW_EF_SEARCH',      50))
RETURN_TOP_K   = int(os.getenv('RETURN_TOP_K',        50))
# ——————————

def fetch_vectors():
    """从 pgvector 表拉取所有 (id, emb)"""
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"SELECT id, {VECTOR_COLUMN} FROM {TABLE_NAME};")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    ids  = [int(r[0]) for r in rows]
    data = np.vstack([np.array(r[1], dtype=np.float32) for r in rows])
    return ids, data


def build_index(data):
    """用 Faiss 构建 HNSW 索引"""
    index = faiss.IndexHNSWFlat(DIM, M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch       = EF_SEARCH
    index.add(data)
    return index


def compute_in_degrees(index, ids):
    """
    利用 HNSW 的 offsets 和 neighbors，统计 Base 层入度：
      - offsets 定位 Base 层范围；
      - neighbors 包含邻居 ID 和 -1 填充；
      - 过滤掉 -1 然后 bincount。
    """
    hnsw      = index.hnsw
    neighbors = faiss.vector_to_array(hnsw.neighbors).astype(np.int32)
    offsets   = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
    num       = len(ids)
    base_end  = int(offsets[num])      # Base 层结束边界
    base_nei  = neighbors[:base_end]
    # 过滤掉 -1 占位符
    valid     = base_nei[base_nei >= 0]
    counts    = np.bincount(valid, minlength=num)
    # 映射回原始 db id
    in_degree = {db_id: int(counts[pos]) for pos, db_id in enumerate(ids)}
    return in_degree


def main():
    print("Fetching vectors from DB...")
    ids, data = fetch_vectors()
    print(f"  → Retrieved {len(ids)} vectors.")

    print("Building Faiss HNSW index in-memory...")
    idx = build_index(data)
    print("  → Index built.")

    print("Computing in-degrees (Base layer)...")
    in_degs = compute_in_degrees(idx, ids)

    low = sorted(in_degs.items(), key=lambda x: x[1])[:RETURN_TOP_K]
    print(f"Top {RETURN_TOP_K} low in-degree nodes (id, in_degree):")
    for oid, deg in low:
        print(f"  ID={oid}, in_degree={deg}")

    print("Done.")

if __name__ == '__main__':
    main()
