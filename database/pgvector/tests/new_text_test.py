#!/usr/bin/env python3
"""
db_watermark.py

对数据库中低入度向量批量嵌入并提取水印：
- 原始向量保存在 emb 列，嵌入后写入 emb_wm 列
- 提取时用 emb 构建索引选 low_ids，再从 emb_wm 拉出向量解码
"""
import os
import psycopg2
import psycopg2.extras
import numpy as np
import faiss
import torch
from pgvector.psycopg2 import register_vector
from core.watermark import VectorWatermark

# —— 配置区 —— #
DB_PARAMS      = {
    'host':     os.getenv('PG_HOST',     'localhost'),
    'port':     int(os.getenv('PG_PORT', 5432)),
    'dbname':   os.getenv('PG_DATABASE', 'test'),
    'user':     os.getenv('PG_USER',     'postgres'),
    'password': os.getenv('PG_PASSWORD', 'ysj'),
}
TABLE_NAME      = 'items'
COL_ORIG        = 'emb'      # 原始向量列
COL_WM          = 'emb_wm'   # 水印向量列
DIM             = 384
M               = int(os.getenv('HNSW_M', 16))
EF_CONSTRUCTION = int(os.getenv('HNSW_EF_CONSTRUCTION', 200))
EF_SEARCH       = int(os.getenv('HNSW_EF_SEARCH', 50))
MODEL_PATH      = os.getenv('WM_MODEL_PATH', 'results/vector_val/best.pt')

MSG_LEN       = 24    # 4 idx +4 CRC +16 payload
BLOCK_PAYLOAD = 16
BLOCK_COUNT   = 16
TOTAL_VECS    = 1600
# ————————————— #


def crc4(bits4):
    reg = 0
    for bit in bits4:
        reg ^= (bit << 3)
        for _ in range(4):
            if reg & 0x8:
                reg = ((reg << 1) & 0xF) ^ 0x3
            else:
                reg = (reg << 1) & 0xF
    return [(reg >> i) & 1 for i in reversed(range(4))]


def text_to_bits(s: str) -> np.ndarray:
    b = s.encode('utf-8')
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def bits_to_text(bits: np.ndarray) -> str:
    by = np.packbits(bits.astype(np.uint8))
    try:
        return by.tobytes().decode('utf-8')
    except:
        return by.tobytes().decode('utf-8', errors='ignore')


def fetch_vectors(col: str):
    """
    从指定列拉出所有 (id, vector)
    """
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"SELECT id, {col} FROM {TABLE_NAME};")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    ids  = [int(r[0]) for r in rows]
    data = np.vstack([np.array(r[1], dtype=np.float32) for r in rows])
    return ids, data


def update_wm_vectors(ids: list, stegos: list):
    """
    批量把 stego 向量写回 emb_wm 列
    """
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur = conn.cursor()
    records = [(st.tolist(), int(id_)) for st, id_ in zip(stegos, ids)]
    psycopg2.extras.execute_batch(
        cur,
        f"UPDATE {TABLE_NAME} SET {COL_WM} = %s WHERE id = %s",
        records
    )
    conn.commit()
    cur.close()
    conn.close()


def build_hnsw_index(data: np.ndarray):
    idx = faiss.IndexHNSWFlat(DIM, M)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    idx.hnsw.efSearch       = EF_SEARCH
    idx.add(data)
    return idx


def compute_in_degrees(idx, ids: list):
    neigh   = faiss.vector_to_array(idx.hnsw.neighbors).astype(np.int32)
    offsets = faiss.vector_to_array(idx.hnsw.offsets).astype(np.int64)
    end     = int(offsets[len(ids)])
    valid   = neigh[:end]
    valid   = valid[valid >= 0]
    cnts    = np.bincount(valid, minlength=len(ids))
    return { dbid: int(cnts[i]) for i, dbid in enumerate(ids) }


def select_low_degree_ids(ids: list, in_degs: dict, k: int):
    pairs = sorted(in_degs.items(), key=lambda x: x[1])
    return [pid for pid, _ in pairs[:k]]


def partition_message(orig_str: str):
    assert len(orig_str) == BLOCK_COUNT * 2
    bits = text_to_bits(orig_str)
    return [
        bits[i*BLOCK_PAYLOAD:(i+1)*BLOCK_PAYLOAD]
        for i in range(BLOCK_COUNT)
    ]


def embed_into_db(low_ids: list, data: np.ndarray, chunks: list, wm: VectorWatermark):
    """
    按 low_ids 顺序对原始列 data 做水印嵌入，逆归一化后写入 emb_wm
    """
    # 原始向量取用 emb
    idx_map  = {id_: i for i, id_ in enumerate(low_ids)}
    sel_data = data[[idx_map[id_] for id_ in low_ids]]
    norms    = np.linalg.norm(sel_data, axis=1) + 1e-8

    stegos = []
    device = wm.device

    for i, oid in enumerate(low_ids):
        blk      = i % BLOCK_COUNT  # 也可以随机分配或平均分配
        idx_bits = [(blk >> (3-b)) & 1 for b in range(4)]
        crc_bits = crc4(idx_bits)
        payload  = chunks[blk].tolist()
        msg_bits = idx_bits + crc_bits + payload

        vec       = sel_data[i]
        cover     = torch.tensor(vec, device=device)
        cover_norm= cover / (cover.norm(p=2) + 1e-8)

        msg_t     = torch.tensor([msg_bits], dtype=torch.float32, device=device)
        stego_n, _= wm.encode(cover_norm.unsqueeze(0), message=msg_t)
        stego_n   = stego_n.squeeze(0)
        stego     = (stego_n * norms[i]).cpu().numpy().astype(np.float32)
        stegos.append(stego)

    update_wm_vectors(low_ids, stegos)


def extract_from_db(low_ids: list, wm: VectorWatermark):
    """
    从 emb_wm 列提取水印，按解码出的索引位分组
    """
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur  = conn.cursor()
    cur.execute(
        f"SELECT id, {COL_WM} FROM {TABLE_NAME} WHERE id = ANY(%s)",
        (low_ids,)
    )
    rows = cur.fetchall()
    cur.close(); conn.close()

    rec_payloads = {b: [] for b in range(BLOCK_COUNT)}
    device = wm.device

    for oid, vec in rows:
        vec        = np.array(vec, dtype=np.float32)
        cover      = torch.tensor(vec, device=device)
        cover_norm = cover / (cover.norm(p=2) + 1e-8)

        rec_bits = wm.decode(cover_norm.unsqueeze(0)) \
                     .squeeze(0).cpu().numpy().astype(int).tolist()
        idx_bits, crc_bits = rec_bits[:4], rec_bits[4:8]
        payload = rec_bits[8:]

        if crc_bits != crc4(idx_bits):
            continue
        blk = sum(idx_bits[i] << (3 - i) for i in range(4))
        if not (0 <= blk < BLOCK_COUNT):
            continue

        rec_payloads[blk].append(np.array(payload, dtype=int))

    return rec_payloads


def recover_text(rec_payloads: dict, orig_str: str):
    print("\nBlock-by-block recovery:")
    recovered = []
    for b in range(BLOCK_COUNT):
        samples = rec_payloads[b]
        orig_txt= orig_str[2*b:2*b+2]
        if not samples:
            print(f"Block {b:2d}: original='{orig_txt}', NO SAMPLES")
            recovered.append("??")
            continue
        mat  = np.stack(samples, axis=0)
        avg  = mat.mean(axis=0)
        bits = (avg > 0.5).astype(int)
        txt  = bits_to_text(bits)
        print(f"Block {b:2d}: original='{orig_txt}', recovered='{txt}' [{len(samples)}]")
        recovered.append(txt)
    final = "".join(recovered)
    print(f"\nFinal: {final}\n")
    return final


def main():
    # 1) 准备文本 & 分块
    orig_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEF"  # 32 字符
    chunks   = partition_message(orig_str)

    # 2) 拉原始列 emb，建索引，选 low_ids
    ids, data = fetch_vectors(COL_ORIG)
    idx       = build_hnsw_index(data)
    in_deg    = compute_in_degrees(idx, ids)
    low_ids   = select_low_degree_ids(ids, in_deg, TOTAL_VECS)
    print(f"Selected {len(low_ids)} low-degree vectors.\n")

    # 3) 嵌入并写入 emb_wm
    wm = VectorWatermark(vec_dim=DIM, msg_len=MSG_LEN, model_path=MODEL_PATH)
    print("Embedding into new column...")
    embed_into_db(low_ids, data, chunks, wm)

    # —— 新增：打印所有 1600 个 ID —— #
    print("Low-degree IDs:")

    print(", ".join(str(i) for i in low_ids))
    print()

    # 4) 提取 & 恢复
    print("Extracting from new column...")
    rec = extract_from_db(low_ids, wm)
    recover_text(rec, orig_str)


if __name__ == "__main__":
    main()
