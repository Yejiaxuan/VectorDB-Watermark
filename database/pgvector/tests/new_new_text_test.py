#!/usr/bin/env python3
"""
db_watermark.py - 单列水印版本

对数据库中低入度向量批量嵌入并提取水印：
- 直接在原始emb列上嵌入水印
- 通过保存低入度向量ID列表确保每次提取使用相同向量集合
- 支持自定义主键列名
"""
import os
import json
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
ID_COL          = 'id'       # 主键列名称
EMB_COL         = 'emb'      # 向量列名称
DIM             = 384
M               = int(os.getenv('HNSW_M', 16))
EF_CONSTRUCTION = int(os.getenv('HNSW_EF_CONSTRUCTION', 200))
EF_SEARCH       = int(os.getenv('HNSW_EF_SEARCH', 50))
MODEL_PATH      = os.getenv('WM_MODEL_PATH', 'results/vector_val/best.pt')

# ID列表文件路径
IDS_FILE       = 'low_degree_ids.json' # 低入度ID列表文件

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


def fetch_vectors():
    """
    从向量列拉出所有 (id, vector)
    """
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur = conn.cursor()
    # 添加ORDER BY确保顺序稳定
    cur.execute(f"SELECT {ID_COL}, {EMB_COL} FROM {TABLE_NAME} ORDER BY {ID_COL};")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    ids  = [int(r[0]) for r in rows]
    data = np.vstack([np.array(r[1], dtype=np.float32) for r in rows])
    return ids, data


def update_vectors(ids: list, stegos: list):
    """
    批量把嵌入水印的向量写回数据库
    """
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur = conn.cursor()
    records = [(st.tolist(), int(id_)) for st, id_ in zip(stegos, ids)]
    psycopg2.extras.execute_batch(
        cur,
        f"UPDATE {TABLE_NAME} SET {EMB_COL} = %s WHERE {ID_COL} = %s",
        records
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"成功更新 {len(records)} 条向量记录")


def build_hnsw_index(data: np.ndarray):
    """
    构建HNSW索引（在嵌入水印时使用）
    """
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
    # 确保结果稳定：入度相同时按ID排序
    pairs = sorted(in_degs.items(), key=lambda x: (x[1], x[0]))
    return [pid for pid, _ in pairs[:k]]


def save_low_degree_ids(low_ids, file_path=IDS_FILE):
    """
    将低入度向量ID列表保存到文件
    """
    print(f"保存低入度向量ID列表到 {file_path}")
    with open(file_path, 'w') as f:
        json.dump(low_ids, f)


def load_low_degree_ids(file_path=IDS_FILE):
    """
    从文件加载低入度向量ID列表
    """
    print(f"从 {file_path} 加载低入度向量ID列表")
    with open(file_path, 'r') as f:
        return json.load(f)


def partition_message(orig_str: str):
    assert len(orig_str) == BLOCK_COUNT * 2
    bits = text_to_bits(orig_str)
    return [
        bits[i*BLOCK_PAYLOAD:(i+1)*BLOCK_PAYLOAD]
        for i in range(BLOCK_COUNT)
    ]


def embed_into_db(low_ids: list, data: np.ndarray, chunks: list, wm: VectorWatermark):
    """
    按 low_ids 顺序对原始列中的向量嵌入水印，并写回数据库
    """
    # 获取低入度向量数据
    idx_map  = {id_: i for i, id_ in enumerate(low_ids)}
    sel_data = data[[idx_map[id_] for id_ in low_ids]]
    norms    = np.linalg.norm(sel_data, axis=1) + 1e-8

    stegos = []
    device = wm.device

    for i, oid in enumerate(low_ids):
        blk      = i % BLOCK_COUNT  # 块索引
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

    # 更新向量到数据库
    update_vectors(low_ids, stegos)


def extract_from_db(low_ids: list, wm: VectorWatermark):
    """
    从数据库中提取水印向量，按解码出的索引位分组
    """
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur  = conn.cursor()
    # 添加ORDER BY确保顺序一致
    cur.execute(
        f"SELECT {ID_COL}, {EMB_COL} FROM {TABLE_NAME} WHERE {ID_COL} = ANY(%s) ORDER BY {ID_COL}",
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
    print("\n区块恢复详情:")
    recovered = []
    for b in range(BLOCK_COUNT):
        samples = rec_payloads[b]
        orig_txt= orig_str[2*b:2*b+2]
        if not samples:
            print(f"区块 {b:2d}: 原文='{orig_txt}', 无样本")
            recovered.append("??")
            continue
        mat  = np.stack(samples, axis=0)
        avg  = mat.mean(axis=0)
        bits = (avg > 0.5).astype(int)
        txt  = bits_to_text(bits)
        print(f"区块 {b:2d}: 原文='{orig_txt}', 恢复='{txt}' [{len(samples)}个样本]")
        recovered.append(txt)
    final = "".join(recovered)
    print(f"\n最终结果: {final}\n")
    return final


def backup_vectors(low_ids, file_path='original_vectors_backup.npz'):
    """
    备份原始向量（可选功能）
    """
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(
        f"SELECT {ID_COL}, {EMB_COL} FROM {TABLE_NAME} WHERE {ID_COL} = ANY(%s) ORDER BY {ID_COL}",
        (low_ids,)
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    
    backup_data = {}
    for id_, vec in rows:
        backup_data[id_] = np.array(vec, dtype=np.float32)
    
    np.savez(file_path, **backup_data)
    print(f"已备份 {len(backup_data)} 个原始向量到 {file_path}")


def embed_watermark():
    """嵌入水印流程"""
    # 1) 准备文本 & 分块
    print("准备消息数据...")
    orig_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEF"  # 32 字符
    chunks   = partition_message(orig_str)

    # 2) 拉原始列向量
    print("获取向量数据...")
    ids, data = fetch_vectors()
    
    # 直接重新构建索引和生成ID列表
    print("构建HNSW索引...")
    idx = build_hnsw_index(data.copy())  # 复制数据以避免可能的修改
    
    print("计算向量入度...")
    in_deg = compute_in_degrees(idx, ids)
    
    print("选择低入度向量...")
    low_ids = select_low_degree_ids(ids, in_deg, TOTAL_VECS)
    
    # 保存ID列表到文件(覆盖现有文件)
    save_low_degree_ids(low_ids)
    
    # 可选：备份原始向量
    backup_choice = input("是否备份原始向量? (y/n): ").lower()
    if backup_choice == 'y':
        backup_vectors(low_ids)
    
    print(f"已选择并保存 {len(low_ids)} 个低入度向量ID")

    # 3) 嵌入并写回原始列
    print("\n准备嵌入水印...")
    wm = VectorWatermark(vec_dim=DIM, msg_len=MSG_LEN, model_path=MODEL_PATH)
    print("嵌入水印中...")
    embed_into_db(low_ids, data, chunks, wm)
    print("水印嵌入完成")

    return orig_str


def extract_watermark():
    """提取水印流程"""
    # 加载低入度向量ID列表
    if not os.path.exists(IDS_FILE):
        print(f"错误: 找不到ID列表文件 {IDS_FILE}")
        return
    
    low_ids = load_low_degree_ids()
    print(f"已加载 {len(low_ids)} 个向量ID")
    
    # 初始化水印模型
    print("初始化水印模型...")
    wm = VectorWatermark(vec_dim=DIM, msg_len=MSG_LEN, model_path=MODEL_PATH)
    
    # 提取水印
    print("从数据库提取水印向量...")
    rec = extract_from_db(low_ids, wm)
    
    # 恢复消息
    print("恢复水印消息...")
    orig_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEF"  # 原始消息，用于比较
    return recover_text(rec, orig_str)


def main():
    print("向量水印系统 (单列版)\n")
    print("请选择操作:")
    print("1. 嵌入水印 (构建索引并保存ID列表)")
    print("2. 提取水印 (使用已保存的ID列表)")
    choice = input("选择 [1/2]: ")
    
    if choice == '1':
        embed_watermark()
    elif choice == '2':
        extract_watermark()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()