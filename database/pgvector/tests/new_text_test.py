"""
batch_text_test.py

批量文本水印测试：
- 一条 32 字母字符串 → 切 16 段 ×2 字母
- 选 1600 条低入度向量，随机均匀分配到 16 段
- 每段用 4bit 索引 + 4bit CRC-4 + 16bit 载荷 = 24bit 嵌入/提取
- 丢弃 CRC 校验失败的提取
- 对同一段所有提取载荷做多数投票，恢复 2 字母
- 最终拼接回 32 字母
"""
import os
import sys
from pathlib import Path

# 把项目根加入 path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import psycopg2
import numpy as np
import faiss
import torch
from pgvector.psycopg2 import register_vector

from vector_watermark import VectorWatermark

# —— 配置区 —— #
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

MSG_LEN         = 24
BLOCK_PAYLOAD   = 16
BLOCK_COUNT     = 16
TOTAL_VECS      = 1600
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
    byte_arr = np.packbits(bits.astype(np.uint8))
    try:
        return byte_arr.tobytes().decode('utf-8')
    except:
        return byte_arr.tobytes().decode('utf-8', errors='ignore')

def fetch_vectors():
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"SELECT id, {VECTOR_COLUMN} FROM {TABLE_NAME};")
    rows = cur.fetchall()
    cur.close(); conn.close()
    ids  = [int(r[0]) for r in rows]
    data = np.vstack([np.array(r[1], dtype=np.float32) for r in rows])
    return ids, data

def build_index(data):
    idx = faiss.IndexHNSWFlat(DIM, M)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    idx.hnsw.efSearch       = EF_SEARCH
    idx.add(data)
    return idx

def compute_in_degrees(idx, ids):
    hnsw  = idx.hnsw
    neigh = faiss.vector_to_array(hnsw.neighbors).astype(np.int32)
    offs  = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
    end   = int(offs[len(ids)])
    valid = neigh[:end]
    valid = valid[valid >= 0]
    cnts  = np.bincount(valid, minlength=len(ids))
    return {dbid: int(cnts[i]) for i, dbid in enumerate(ids)}

def main():
    # 1) 原始文本 & 切块
    orig_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEF"
    assert len(orig_str) == 32, "请确保字符串长度正好 32"
    orig_bits = text_to_bits(orig_str)  # 256 bits
    chunks = [ orig_bits[i*BLOCK_PAYLOAD:(i+1)*BLOCK_PAYLOAD]
               for i in range(BLOCK_COUNT) ]

    # 2) 取向量、重索引、选入度最低
    print("Fetching vectors & building index...")
    ids, data = fetch_vectors()
    idx       = build_index(data)
    in_degs   = compute_in_degrees(idx, ids)
    low_ids   = [oid for oid,_ in sorted(in_degs.items(), key=lambda x:x[1])][:TOTAL_VECS]
    print(f"Selected {TOTAL_VECS} low-in-degree vectors.\n")

    # 3) 随机分配到 16 块
    per_block   = TOTAL_VECS // BLOCK_COUNT
    assignments = [k for k in range(BLOCK_COUNT) for _ in range(per_block)]
    np.random.shuffle(assignments)

    # 4) 嵌入+提取
    wm = VectorWatermark(vec_dim=DIM, msg_len=MSG_LEN, model_path=MODEL_PATH)
    rec_payloads = {k: [] for k in range(BLOCK_COUNT)}

    for i, oid in enumerate(low_ids):
        blk = assignments[i]
        idx_bits = [(blk >> b) & 1 for b in reversed(range(4))]
        crc_bits = crc4(idx_bits)
        payload  = chunks[blk].tolist()
        msg_bits = idx_bits + crc_bits + payload

        pos   = ids.index(oid)
        cover = torch.from_numpy(data[pos]).float().to(wm.device)
        cover = cover / (cover.norm(p=2) + 1e-8)
        msg_t = torch.tensor([msg_bits], dtype=torch.float32).to(wm.device)
        stego, _ = wm.encode(cover.unsqueeze(0), message=msg_t)
        rec = wm.decode(stego).squeeze(0).cpu().numpy().astype(int).tolist()

        rec_idx = rec[0:4]; rec_crc = rec[4:8]; rec_pl = rec[8:24]
        if rec_crc != crc4(rec_idx):
            continue
        k2 = sum(rec_idx[b] << (3-b) for b in range(4))
        if k2 != blk:
            continue
        rec_payloads[blk].append(np.array(rec_pl, dtype=int))

    # —— 下面这段只执行一次 —— #
    print("\nBlock-by-block recovery:")
    recovered = []
    for k in range(BLOCK_COUNT):
        raws = rec_payloads[k]
        orig_txt = orig_str[2*k:2*k+2]
        if not raws:
            print(f"  Block {k:2d}: original='{orig_txt}', NO VALID SAMPLES")
            recovered.append("??")
            continue
        mat  = np.stack(raws, axis=0)
        avg  = mat.mean(axis=0)
        bits = (avg > 0.5).astype(int)
        txt  = bits_to_text(bits)
        print(f"  Block {k:2d}: original='{orig_txt}', recovered='{txt}' [{len(raws)} samples]")
        recovered.append(txt)

    final = "".join(recovered)
    print(f"\nFinal recovered string: {final}")

if __name__ == "__main__":
    main()
