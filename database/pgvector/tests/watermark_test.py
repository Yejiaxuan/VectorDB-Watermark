#!/usr/bin/env python3
"""
embed_extract_low_in_degree.py

对入度最低的 50 个节点进行水印嵌入与提取演示。
暂不修改数据库，仅打印原始向量、带水印向量、待嵌入消息、提取消息和 BER。
"""
import os, sys
# 假设脚本在 pgvector/ 下面，父目录就是项目根
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
import psycopg2
import numpy as np
import faiss
import torch
from pgvector.psycopg2 import register_vector

# 引入水印处理类
from core.watermark import VectorWatermark

# —— 配置区 ——
DB_PARAMS = {
    'host':     os.getenv('PG_HOST',     'localhost'),
    'port':     int(os.getenv('PG_PORT', 5432)),
    'dbname':   os.getenv('PG_DATABASE','test'),
    'user':     os.getenv('PG_USER',    'postgres'),
    'password': os.getenv('PG_PASSWORD','ysj'),
}
TABLE_NAME       = 'items'
VECTOR_COLUMN    = 'emb'
DIM              = 384
M                = int(os.getenv('HNSW_M',             16))
EF_CONSTRUCTION  = int(os.getenv('HNSW_EF_CONSTRUCTION',200))
EF_SEARCH        = int(os.getenv('HNSW_EF_SEARCH',      50))
MSG_LEN          = int(os.getenv('WM_MSG_LEN',         96))
MODEL_PATH       = os.getenv('WM_MODEL_PATH', 'results/vector_val/best.pt')
RETURN_TOP_K     = 50
# —————————————

def fetch_vectors():
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


def build_hnsw_index(data):
    index = faiss.IndexHNSWFlat(DIM, M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch       = EF_SEARCH
    index.add(data)
    return index


def compute_in_degrees(index, ids):
    hnsw      = index.hnsw
    neighbors = faiss.vector_to_array(hnsw.neighbors).astype(np.int32)
    offsets   = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
    num       = len(ids)
    base_end  = int(offsets[num])
    base_nei  = neighbors[:base_end]
    valid     = base_nei[base_nei >= 0]
    counts    = np.bincount(valid, minlength=num)
    return {db_id: int(counts[pos]) for pos, db_id in enumerate(ids)}


def main():
    # 拉取向量 & 构建索引
    print("Fetching vectors from DB...")
    ids, data = fetch_vectors()
    print(f"Retrieved {len(ids)} vectors.")
    print("Building HNSW index in-memory...")
    idx = build_hnsw_index(data)
    print("Index built.")

    # 计算入度并选出最小 50
    print("Computing in-degrees...")
    in_degs = compute_in_degrees(idx, ids)
    low_ids = [oid for oid, _ in sorted(in_degs.items(), key=lambda x: x[1])[:RETURN_TOP_K]]
    print(f"Top {RETURN_TOP_K} low in-degree IDs: {low_ids}\n")

    # 初始化水印处理器
    wm = VectorWatermark(vec_dim=DIM, msg_len=MSG_LEN, model_path=MODEL_PATH)

    # 对每个低入度向量做嵌入与提取
    for oid in low_ids:
        pos = ids.index(oid)
        cover = data[pos]  # 384-d vector

        # 转换为 PyTorch 张量
        cover_tensor = torch.from_numpy(cover).float()

        # 归一化
        norm = torch.norm(cover_tensor, p=2, dim=-1, keepdim=True)
        cover_tensor = cover_tensor / (norm + 1e-8)

        # 嵌入水印，随机消息
        stego_tensor, msg_bits = wm.encode(cover_tensor, random_msg=True)
        stego_np = stego_tensor.cpu().numpy().squeeze()

        # 提取水印
        rec_bits = wm.decode(stego_tensor)

        # 计算 BER 与 MSE
        ber = wm.compute_ber(msg_bits, rec_bits)
        # 将PyTorch张量转换为NumPy数组以进行MSE计算
        mse = float(((stego_np - cover_tensor.cpu().numpy().squeeze())**2).mean())

        # 添加这里：逆归一化，恢复原始尺度
        stego_np = stego_np * norm.cpu().numpy().squeeze()

        # 打印信息
        print(f"ID={oid}: in_degree={in_degs[oid]}")
        print(f"  原始消息 bits= {msg_bits.tolist()}")
        print(f"  提取消息 bits= {rec_bits.tolist()}")
        print(f"  cover[:5]= {cover[:5].tolist()}...")
        print(f"  stego[:5]= {stego_np[:5].tolist()}...")
        print(f"  BER={ber:.2%}, MSE={mse:.6f}\n")

if __name__ == '__main__':
    main()
