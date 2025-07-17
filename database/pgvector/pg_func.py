"""
向量水印系统核心函数库 (API版)

提供向量水印嵌入和提取的关键功能：
- 支持参数化数据库连接和表结构
- 支持自定义主键列名和向量列名
- 通过文件保存低入度ID列表确保嵌入/提取一致性
"""
import os
import json
import psycopg2
import psycopg2.extras
import numpy as np
import faiss
import torch
from pgvector.psycopg2 import register_vector
from algorithms.deep_learning.watermark import VectorWatermark
from configs.config import Config

# —— 从配置文件获取参数 —— #
M = Config.HNSW_M
EF_CONSTRUCTION = Config.HNSW_EF_CONSTRUCTION
EF_SEARCH = Config.HNSW_EF_SEARCH

# —— 水印参数 —— #
MSG_LEN = Config.MSG_LEN
BLOCK_PAYLOAD = Config.BLOCK_PAYLOAD
BLOCK_COUNT = Config.BLOCK_COUNT
DEFAULT_EMBED_RATE = Config.DEFAULT_EMBED_RATE


def crc4(bits4):
    """计算4位CRC校验和"""
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
    """文本转比特序列"""
    b = s.encode('utf-8')
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def bits_to_text(bits: np.ndarray) -> str:
    """比特序列转文本"""
    by = np.packbits(bits.astype(np.uint8))
    try:
        return by.tobytes().decode('utf-8')
    except:
        return by.tobytes().decode('utf-8', errors='ignore')


def fetch_vectors(db_params, table_name, id_col, emb_col):
    """
    从向量列拉出所有 (id, vector)
    """
    conn = psycopg2.connect(**db_params)
    register_vector(conn)
    cur = conn.cursor()
    # 添加ORDER BY确保顺序稳定
    cur.execute(f"SELECT {id_col}, {emb_col} FROM {table_name} ORDER BY {id_col};")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    ids = [int(r[0]) for r in rows]
    data = np.vstack([np.array(r[1], dtype=np.float32) for r in rows])
    return ids, data


def update_vectors(db_params, table_name, id_col, emb_col, ids, stegos):
    """
    批量把嵌入水印的向量写回数据库
    """
    conn = psycopg2.connect(**db_params)
    register_vector(conn)
    cur = conn.cursor()
    records = [(st.tolist(), int(id_)) for st, id_ in zip(stegos, ids)]
    psycopg2.extras.execute_batch(
        cur,
        f"UPDATE {table_name} SET {emb_col} = %s WHERE {id_col} = %s",
        records
    )
    conn.commit()
    cur.close()
    conn.close()
    return len(records)


def build_hnsw_index(data: np.ndarray, vec_dim: int):
    """
    构建HNSW索引（在嵌入水印时使用）
    """
    idx = faiss.IndexHNSWFlat(vec_dim, M)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    idx.hnsw.efSearch = EF_SEARCH
    # 确保数据类型为float32
    data_float32 = data.astype(np.float32)
    idx.add(data_float32)
    return idx


def compute_in_degrees(idx, ids: list):
    """计算索引中每个向量的入度"""
    neigh = faiss.vector_to_array(idx.hnsw.neighbors).astype(np.int32)
    offsets = faiss.vector_to_array(idx.hnsw.offsets).astype(np.int64)
    end = int(offsets[len(ids)])
    valid = neigh[:end]
    valid = valid[valid >= 0]
    cnts = np.bincount(valid, minlength=len(ids))
    return {dbid: int(cnts[i]) for i, dbid in enumerate(ids)}


def select_low_degree_ids_by_rate(ids: list, in_degs: dict, embed_rate: float):
    """
    按水印嵌入率从低入度向量中选择指定数量的向量ID
    
    Args:
        ids: 所有向量ID列表
        in_degs: 每个ID的入度字典
        embed_rate: 水印嵌入率（0-1之间的浮点数）
        
    Returns:
        选中的向量ID列表，按入度从低到高排序
    """
    # 计算需要选择的向量数量
    target_count = int(len(ids) * embed_rate)
    if target_count < BLOCK_COUNT:
        # 如果目标数量少于块数，至少保证有足够的向量
        target_count = BLOCK_COUNT

    # 按入度排序所有向量（入度相同时按ID排序确保稳定性）
    sorted_pairs = sorted([(id_, in_degs[id_]) for id_ in ids], key=lambda x: (x[1], x[0]))

    # 选择前target_count个向量
    selected = [pid for pid, _ in sorted_pairs[:target_count]]

    return selected


def select_low_degree_ids(ids: list, in_degs: dict, k=None):
    """
    选择入度≤指定阈值的向量ID（保留旧接口以保持兼容性）
    """
    # 兼容旧版本的固定阈值方式，这里设置一个较大的默认值
    max_degree = k if k is not None else 10
    filtered = [(id_, deg) for id_, deg in in_degs.items() if deg <= max_degree]
    # 确保结果稳定：入度相同时按ID排序
    pairs = sorted(filtered, key=lambda x: (x[1], x[0]))
    return [pid for pid, _ in pairs]


def save_low_degree_ids(low_ids, file_path):
    """将低入度向量ID列表保存到文件"""
    with open(file_path, 'w') as f:
        json.dump(low_ids, f)
    return len(low_ids)


def load_low_degree_ids(file_path):
    """从文件加载低入度向量ID列表"""
    with open(file_path, 'r') as f:
        return json.load(f)


def partition_message(orig_str: str):
    """将文本消息分割为多个块"""
    assert len(orig_str) == BLOCK_COUNT * 2
    bits = text_to_bits(orig_str)
    return [
        bits[i * BLOCK_PAYLOAD:(i + 1) * BLOCK_PAYLOAD]
        for i in range(BLOCK_COUNT)
    ]


def embed_into_db(low_ids: list, data: np.ndarray, chunks: list, wm: VectorWatermark, db_params, table_name, id_col,
                  emb_col):
    """
    按 low_ids 顺序对原始列中的向量嵌入水印，并写回数据库
    使用循环分配方式，确保每个块都有足够的向量
    """
    # 获取低入度向量数据
    idx_map = {id_: i for i, id_ in enumerate(low_ids)}
    sel_data = data[[idx_map[id_] for id_ in low_ids]]
    norms = np.linalg.norm(sel_data, axis=1) + 1e-8

    stegos = []
    device = wm.device

    # 统计每个块分配了多少向量，用于日志
    block_counters = [0] * BLOCK_COUNT

    for i, oid in enumerate(low_ids):
        # 循环分配块索引 (0,1,2,...,15,0,1,2,...)
        blk = i % BLOCK_COUNT
        block_counters[blk] += 1

        # 构造索引位和CRC
        idx_bits = [(blk >> (3 - b)) & 1 for b in range(4)]
        crc_bits = crc4(idx_bits)
        payload = chunks[blk].tolist()
        msg_bits = idx_bits + crc_bits + payload

        # 嵌入水印
        vec = sel_data[i]
        cover = torch.tensor(vec, device=device).unsqueeze(0)

        msg_t = torch.tensor([msg_bits], dtype=torch.float32, device=device)
        stego, _ = wm.encode(cover, message=msg_t)
        stego = stego.squeeze(0).cpu().numpy().astype(np.float32)
        stegos.append(stego)

    # 打印每个区块分配的向量数量
    print(f"Block distribution: {block_counters}")

    # 更新向量到数据库
    return update_vectors(db_params, table_name, id_col, emb_col, low_ids, stegos)


def extract_from_db(db_params, table_name, id_col, emb_col, low_ids: list, wm: VectorWatermark):
    """
    从数据库中提取水印向量，按解码出的索引位分组
    """
    conn = psycopg2.connect(**db_params)
    register_vector(conn)
    cur = conn.cursor()
    # 添加ORDER BY确保顺序一致
    cur.execute(
        f"SELECT {id_col}, {emb_col} FROM {table_name} WHERE {id_col} = ANY(%s) ORDER BY {id_col}",
        (low_ids,)
    )
    rows = cur.fetchall()
    cur.close();
    conn.close()

    rec_payloads = {b: [] for b in range(BLOCK_COUNT)}
    device = wm.device

    for oid, vec in rows:
        vec = np.array(vec, dtype=np.float32)
        cover = torch.tensor(vec, device=device).unsqueeze(0)

        rec_bits = wm.decode(cover) \
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


def backup_vectors(db_params, table_name, id_col, emb_col, low_ids, file_path='original_vectors_backup.npz'):
    """
    备份原始向量
    """
    conn = psycopg2.connect(**db_params)
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(
        f"SELECT {id_col}, {emb_col} FROM {table_name} WHERE {id_col} = ANY(%s) ORDER BY {id_col}",
        (low_ids,)
    )
    rows = cur.fetchall()
    cur.close();
    conn.close()

    backup_data = {}
    for id_, vec in rows:
        backup_data[str(id_)] = np.array(vec, dtype=np.float32)

    np.savez(file_path, **backup_data)
    return len(backup_data)


def embed_watermark(db_params, table_name, id_col, emb_col, message, vec_dim, embed_rate=None, total_vecs=None, ids_file=None):
    """
    嵌入水印流程 - 使用指定嵌入率选择低入度向量，不保存ID文件
    
    Args:
        db_params: 数据库连接参数
        table_name: 表名
        id_col: 主键列名
        emb_col: 向量列名
        message: 水印消息
        embed_rate: 水印嵌入率（0-1之间的浮点数），优先使用此参数
        total_vecs: 使用的向量数量（已弃用，保留兼容性）
        ids_file: ID文件路径（已弃用，保留兼容性）
    """
    # 检查消息长度
    if len(message) != BLOCK_COUNT * 2:
        return {"success": False, "error": f"消息长度必须为 {BLOCK_COUNT * 2} 字符"}

    # 确定嵌入率
    if embed_rate is None:
        embed_rate = DEFAULT_EMBED_RATE

    if not (0 < embed_rate <= 1):
        return {"success": False, "error": "水印嵌入率必须在0和1之间"}

    # 1) 分块消息数据
    chunks = partition_message(message)

    # 2) 拉原始列向量
    try:
        ids, data = fetch_vectors(db_params, table_name, id_col, emb_col)
    except Exception as e:
        return {"success": False, "error": f"获取向量数据失败: {str(e)}"}

    # 3) 构建索引和生成ID列表
    try:
        idx = build_hnsw_index(data.copy(), vec_dim)
        in_deg = compute_in_degrees(idx, ids)

        # 按嵌入率选择低入度向量
        low_ids = select_low_degree_ids_by_rate(ids, in_deg, embed_rate)

        # 如果选出的向量太少，返回错误
        if len(low_ids) < BLOCK_COUNT:
            return {
                "success": False,
                "error": f"按{embed_rate:.1%}嵌入率选择的向量数量({len(low_ids)})少于最小需求({BLOCK_COUNT})"
            }

    except Exception as e:
        return {"success": False, "error": f"生成低入度向量ID失败: {str(e)}"}

    # 4) 嵌入并写回原始列
    try:
        model_path = Config.get_model_path(vec_dim)
        wm = VectorWatermark(vec_dim=vec_dim, msg_len=MSG_LEN, model_path=model_path)
        updated = embed_into_db(low_ids, data, chunks, wm, db_params, table_name, id_col, emb_col)

        # 计算实际嵌入率
        actual_rate = len(low_ids) / len(ids)

        # 返回嵌入结果，不包含文件信息
        return {
            "success": True,
            "message": f"水印嵌入完成，嵌入率{actual_rate:.1%}({len(low_ids)}/{len(ids)})，更新了{updated}个向量",
            "updated": updated,
            "used_vectors": len(low_ids),
            "total_vectors": len(ids),
            "embed_rate": actual_rate
        }
    except Exception as e:
        return {"success": False, "error": f"嵌入水印失败: {str(e)}"}


def extract_watermark(db_params, table_name, id_col, emb_col, vec_dim, embed_rate=None, ids_file=None):
    """
    提取水印流程 - 重新计算低入度节点并使用纯统计方法
    
    Args:
        db_params: 数据库连接参数
        table_name: 表名
        id_col: 主键列名
        emb_col: 向量列名
        embed_rate: 水印嵌入率（0-1之间的浮点数），如果提供则使用此参数重新计算
        ids_file: ID文件路径（已弃用，保留兼容性）
    """

    # 确定嵌入率
    if embed_rate is None:
        embed_rate = DEFAULT_EMBED_RATE

    if not (0 < embed_rate <= 1):
        return {"success": False, "error": "水印嵌入率必须在0和1之间"}

    # 1) 重新获取向量数据并计算低入度节点
    try:
        ids, data = fetch_vectors(db_params, table_name, id_col, emb_col)

        # 重新构建索引并计算入度
        idx = build_hnsw_index(data.copy(), vec_dim)
        in_deg = compute_in_degrees(idx, ids)

        # 按嵌入率选择低入度向量（与嵌入时使用相同策略）
        low_ids = select_low_degree_ids_by_rate(ids, in_deg, embed_rate)

        if len(low_ids) < BLOCK_COUNT:
            return {
                "success": False,
                "error": f"按{embed_rate:.1%}嵌入率选择的向量数量({len(low_ids)})少于最小需求({BLOCK_COUNT})"
            }

        print(f"提取时按{embed_rate:.1%}嵌入率找到 {len(low_ids)} 个低入度节点")

    except Exception as e:
        return {"success": False, "error": f"获取向量数据或计算入度失败: {str(e)}"}

    # 2) 提取水印 - 对所有低入度节点进行解码
    try:
        model_path = Config.get_model_path(vec_dim)
        wm = VectorWatermark(vec_dim=vec_dim, msg_len=MSG_LEN, model_path=model_path)

        # 连接数据库获取低入度向量
        conn = psycopg2.connect(**db_params)
        register_vector(conn)
        cur = conn.cursor()
        cur.execute(
            f"SELECT {id_col}, {emb_col} FROM {table_name} WHERE {id_col} = ANY(%s) ORDER BY {id_col}",
            (low_ids,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # 按块收集有效载荷
        rec_payloads = {b: [] for b in range(BLOCK_COUNT)}
        device = wm.device
        valid_decodes = 0
        total_decodes = 0

        for oid, vec in rows:
            total_decodes += 1
            vec = np.array(vec, dtype=np.float32)
            cover = torch.tensor(vec, device=device).unsqueeze(0)

            rec_bits = wm.decode(cover).squeeze(0).cpu().numpy().astype(int).tolist()
            idx_bits, crc_bits = rec_bits[:4], rec_bits[4:8]
            payload = rec_bits[8:]

            # 验证CRC
            if crc_bits != crc4(idx_bits):
                continue

            # 验证块索引
            blk = sum(idx_bits[i] << (3 - i) for i in range(4))
            if not (0 <= blk < BLOCK_COUNT):
                continue

            # 收集有效载荷
            rec_payloads[blk].append(np.array(payload, dtype=int))
            valid_decodes += 1

        print(f"总解码: {total_decodes}, 有效解码: {valid_decodes}")

        # 3) 恢复消息 - 使用纯统计方法
        from collections import Counter

        recovered = []
        recovered_blocks = 0
        stats_info = []  # 记录每个区块的统计信息

        for b in range(BLOCK_COUNT):
            samples = rec_payloads.get(b, [])
            if not samples:
                recovered.append("??")
                stats_info.append({"block": b, "status": "empty", "samples": 0})
                continue

            recovered_blocks += 1

            # 将每个样本转化为比特串的哈希值用于统计
            bit_strings = []
            for sample in samples:
                # 将浮点样本转化为二进制比特
                bits = (sample > 0.5).astype(int)
                # 将比特数组转化为元组，以便作为Counter的键
                bit_tuple = tuple(bits.tolist())
                bit_strings.append(bit_tuple)

            # 统计比特串的出现频率
            counter = Counter(bit_strings)
            most_common = counter.most_common(1)

            if most_common:  # 确保有结果
                most_common_bits, count = most_common[0]
                # 使用出现频率最高的比特串
                best_bits = np.array(most_common_bits)
                txt = bits_to_text(best_bits)

                # 记录统计信息
                stats_info.append({
                    "block": b,
                    "status": "found",
                    "samples": len(samples),
                    "most_common_count": count,
                    "most_common_percent": round(count / len(samples) * 100, 2)
                })
            else:
                # 如果没有样本，返回占位符
                txt = "??"
                stats_info.append({"block": b, "status": "no_results", "samples": len(samples)})

            recovered.append(txt)

        final = "".join(recovered)
        return {
            "success": True,
            "message": final,
            "blocks": BLOCK_COUNT,
            "recovered": recovered_blocks,
            "method": "recomputed_statistical",
            "total_low_degree_nodes": len(low_ids),
            "valid_decodes": valid_decodes,
            "total_decodes": total_decodes,
            "stats": stats_info  # 添加统计信息以便分析
        }

    except Exception as e:
        return {"success": False, "error": f"提取水印失败: {str(e)}"}
