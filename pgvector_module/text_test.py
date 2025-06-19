#!/usr/bin/env python3
"""
embed_text_watermark_debug.py

示例：将文本 "WaterMark" 嵌入到一批低入度向量，
并添加详细日志以便排查哪一位出错。
"""
import os
import sys
from pathlib import Path
# 添加项目根到 sys.path
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import psycopg2
import numpy as np
import faiss
import torch
from pgvector.psycopg2 import register_vector

from vector_watermark import VectorWatermark

# —— 配置区 ——
DB_PARAMS = {
    'host':     os.getenv('PG_HOST', 'localhost'),
    'port':     int(os.getenv('PG_PORT', 5432)),
    'dbname':   os.getenv('PG_DATABASE', 'test'),
    'user':     os.getenv('PG_USER', 'postgres'),
    'password': os.getenv('PG_PASSWORD', 'ysj'),
}
TABLE_NAME      = 'items'
VECTOR_COLUMN   = 'emb'
DIM             = 384
M               = int(os.getenv('HNSW_M', 16))
EF_CONSTRUCTION = int(os.getenv('HNSW_EF_CONSTRUCTION', 200))
EF_SEARCH       = int(os.getenv('HNSW_EF_SEARCH', 50))
MODEL_PATH      = os.getenv('WM_MODEL_PATH', 'results/vector_val/best.pt')
MSG_LEN         = 96     # 与训练一致
RETURN_TOP_K    = 50    # 扩大检验规模
# ———————————————

def text_to_bits(s: str, msg_len: int) -> torch.Tensor:
    b = s.encode('utf-8')
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    if bits.size < msg_len:
        bits = np.concatenate([bits, np.zeros(msg_len - bits.size, dtype=np.uint8)])
    else:
        bits = bits[:msg_len]
    return torch.from_numpy(bits.astype(np.float32))

def bits_to_text(bits: np.ndarray) -> str:
    arr = bits.astype(np.uint8)
    length = (arr.size // 8) * 8
    byte_arr = np.packbits(arr[:length])
    try:
        return byte_arr.tobytes().decode('utf-8')
    except UnicodeDecodeError:
        return byte_arr.tobytes().decode('utf-8', errors='ignore')


def fetch_vectors():
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"SELECT id, {VECTOR_COLUMN} FROM {TABLE_NAME};")
    rows = cur.fetchall()
    cur.close(); conn.close()
    ids = [int(r[0]) for r in rows]
    data = np.vstack([np.array(r[1], dtype=np.float32) for r in rows])
    return ids, data


def build_index(data):
    idx = faiss.IndexHNSWFlat(DIM, M)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    idx.hnsw.efSearch       = EF_SEARCH
    idx.add(data)
    return idx


def compute_in_degrees(idx, ids):
    hnsw = idx.hnsw
    neigh = faiss.vector_to_array(hnsw.neighbors).astype(np.int32)
    offs  = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
    N = len(ids)
    end = int(offs[N])
    valid = neigh[:end]
    valid = valid[valid >= 0]
    cnts = np.bincount(valid, minlength=N)
    return {db_id: int(cnts[i]) for i, db_id in enumerate(ids)}


def main():
    # 1. 加载向量 & 构建索引
    print("Fetching vectors from DB...")
    ids, data = fetch_vectors()
    idx = build_index(data)
    in_degs = compute_in_degrees(idx, ids)
    low_ids = [oid for oid, _ in sorted(in_degs.items(), key=lambda x: x[1])[:RETURN_TOP_K]]
    print(f"Selected {len(low_ids)} low in-degree IDs for embed/test: {low_ids}\n")

    # 2. 初始化 WM
    wm = VectorWatermark(vec_dim=DIM, msg_len=MSG_LEN, model_path=MODEL_PATH)
    message_str = "OurWaterMark"
    orig_bits = text_to_bits(message_str, MSG_LEN).to(wm.device)
    print("Original text:", message_str)
    print("Original bits:", orig_bits.cpu().numpy().tolist(), "\n")

    # 3. 嵌入 & 提取 并打印日志
    rec_bits_list = []
    for idx_i, oid in enumerate(low_ids):
        pos = ids.index(oid)
        cover = torch.from_numpy(data[pos]).float().to(wm.device)
        cover = cover / (cover.norm(p=2) + 1e-8)
        stego, _ = wm.encode(cover.unsqueeze(0), message=orig_bits)
        rec = wm.decode(stego).squeeze(0)
        rec_cpu = rec.cpu().numpy().astype(int)
        # 计算 per-vector 日志
        mismatches = np.where(rec_cpu != orig_bits.cpu().numpy())[0]
        print(f"[{idx_i}] Vector ID={oid}, mismatches at bit positions: {mismatches.tolist()}")
        print(f"    extracted bits: {rec_cpu.tolist()}\n")
        rec_bits_list.append(rec_cpu)

    # 4. 聚合并打印 avg logs
    all_rec = np.stack(rec_bits_list, axis=0)
    avg_prob = all_rec.mean(axis=0)
    print("Average bit probabilities per position:", np.round(avg_prob, 3).tolist())
    agg_bits = (avg_prob > 0.5).astype(int)
    mismatches_agg = np.where(agg_bits != orig_bits.cpu().numpy())[0]
    print("Aggregated mismatches at positions:", mismatches_agg.tolist())
    rec_text = bits_to_text(agg_bits)
    print("Recovered text:", rec_text)

if __name__ == '__main__':
    main()
