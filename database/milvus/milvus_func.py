"""
Milvus向量水印系统核心函数库

提供向量水印嵌入和提取的关键功能：
- 支持参数化Milvus连接和集合结构
- 支持自定义主键字段名和向量字段名
- 通过文件保存低入度ID列表确保嵌入/提取一致性
"""
import os
import json
import uuid
import numpy as np
import faiss
import torch
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from algorithms.deep_learning.watermark import VectorWatermark
from configs.config import Config

# —— 从配置文件获取参数 —— #
DIM = Config.VEC_DIM
M = Config.HNSW_M
EF_CONSTRUCTION = Config.HNSW_EF_CONSTRUCTION
EF_SEARCH = Config.HNSW_EF_SEARCH
MODEL_PATH = os.getenv('WM_MODEL_PATH', Config.MODEL_PATH)

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


def fetch_vectors(db_params, collection_name, id_field, vector_field):
    """
    从Milvus集合中获取所有 (id, vector)
    使用ID范围查询避免offset+limit限制
    """
    alias = f"fetch_{uuid.uuid4().hex[:8]}"
    try:
        connections.connect(
            alias=alias,
            host=db_params.get("host", "localhost"),
            port=db_params.get("port", 19530)
        )

        collection = Collection(collection_name, using=alias)

        # 确保集合已加载
        collection.load()

        # 先获取ID范围
        # 查询最小和最大ID
        min_result = collection.query(
            expr="",
            output_fields=[id_field],
            limit=1,
            offset=0
        )

        if not min_result:
            return [], np.array([]).reshape(0, DIM)

        # 获取总数量（使用Milvus内置方法）
        total_count = collection.num_entities
        print(f"集合总向量数: {total_count}")

        # 使用ID范围分批查询，避免offset限制
        batch_size = 5000  # 保守的批次大小
        all_results = []

        # 查询所有数据，按ID排序
        current_id = 0

        while len(all_results) < total_count:
            # 使用ID范围查询
            expr = f"{id_field} >= {current_id} and {id_field} < {current_id + batch_size}"

            batch_results = collection.query(
                expr=expr,
                output_fields=[id_field, vector_field],
                limit=batch_size
            )

            if not batch_results:
                # 如果没有结果，尝试下一个范围
                current_id += batch_size
                # 防止无限循环
                if current_id > total_count * 2:
                    break
                continue

            all_results.extend(batch_results)

            # 更新当前ID为已查询的最大ID + 1
            max_id_in_batch = max(result[id_field] for result in batch_results)
            current_id = max_id_in_batch + 1

            print(f"已获取 {len(all_results)}/{total_count} 个向量")

            # 如果这批数据少于batch_size，说明可能接近结束
            if len(batch_results) < batch_size:
                # 继续查询剩余数据
                current_id = max_id_in_batch + 1

        print(f"总共获取了 {len(all_results)} 个向量")

        # 按ID排序确保顺序稳定
        all_results.sort(key=lambda x: x[id_field])

        ids = [int(result[id_field]) for result in all_results]
        vectors = [np.array(result[vector_field], dtype=np.float32) for result in all_results]
        data = np.vstack(vectors) if vectors else np.array([]).reshape(0, DIM)

        return ids, data

    finally:
        connections.disconnect(alias)


def update_vectors(db_params, collection_name, id_field, vector_field, ids, stegos):
    """
    批量把嵌入水印的向量写回Milvus
    """
    alias = f"update_{uuid.uuid4().hex[:8]}"
    try:
        connections.connect(
            alias=alias,
            host=db_params.get("host", "localhost"),
            port=db_params.get("port", 19530)
        )

        collection = Collection(collection_name, using=alias)

        # 分批删除和插入数据以避免Milvus查询限制
        batch_size = 10000  # 设置批次大小，低于16384的限制
        total_updated = 0

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_stegos = stegos[i:i + batch_size]

            # 准备当前批次的数据
            batch_entities = []
            for id_val, stego in zip(batch_ids, batch_stegos):
                batch_entities.append({
                    id_field: int(id_val),
                    vector_field: stego.tolist()
                })

            # 删除当前批次的旧数据
            batch_id_list = list(map(int, batch_ids))
            expr = f"{id_field} in {batch_id_list}"
            collection.delete(expr)

            # 插入当前批次的新数据
            collection.insert(batch_entities)
            total_updated += len(batch_entities)

            print(f"已更新 {total_updated}/{len(ids)} 个向量")

        # 刷新所有更改
        collection.flush()

        return total_updated

    finally:
        connections.disconnect(alias)


def build_hnsw_index(data: np.ndarray):
    """
    构建HNSW索引（在嵌入水印时使用）
    """
    idx = faiss.IndexHNSWFlat(DIM, M)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    idx.hnsw.efSearch = EF_SEARCH
    idx.add(data)
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
    根据嵌入率选择入度最低的向量ID
    
    Args:
        ids: 向量ID列表
        in_degs: 入度字典 {id: degree}
        embed_rate: 水印嵌入率（0-1之间的浮点数）
        
    Returns:
        选择的低入度向量ID列表
    """
    # 计算需要选择的向量数量
    target_count = int(len(ids) * embed_rate)

    # 确保至少选择BLOCK_COUNT个向量
    target_count = max(target_count, BLOCK_COUNT)

    # 按入度排序（入度低的优先），入度相同时按ID排序确保结果稳定
    sorted_pairs = sorted([(id_, in_degs.get(id_, 0)) for id_ in ids], key=lambda x: (x[1], x[0]))

    # 取前target_count个
    selected_pairs = sorted_pairs[:target_count]
    return [pid for pid, _ in selected_pairs]


def select_low_degree_ids(ids: list, in_degs: dict, k=None):
    """
    选择入度≤MAX_IN_DEGREE的向量ID（保留用于向后兼容）
    """
    MAX_IN_DEGREE = 5  # 本地定义以保持兼容性
    # 过滤所有入度小于等于MAX_IN_DEGREE的向量
    filtered = [(id_, deg) for id_, deg in in_degs.items() if deg <= MAX_IN_DEGREE]
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


def embed_into_milvus(low_ids: list, data: np.ndarray, chunks: list, wm: VectorWatermark,
                      db_params, collection_name, id_field, vector_field):
    """
    按 low_ids 顺序对原始集合中的向量嵌入水印，并写回Milvus
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

    # 更新向量到Milvus
    return update_vectors(db_params, collection_name, id_field, vector_field, low_ids, stegos)


def extract_from_milvus(db_params, collection_name, id_field, vector_field, low_ids: list, wm: VectorWatermark):
    """
    从Milvus中提取水印向量，按解码出的索引位分组
    """
    alias = f"extract_{uuid.uuid4().hex[:8]}"
    try:
        connections.connect(
            alias=alias,
            host=db_params.get("host", "localhost"),
            port=db_params.get("port", 19530)
        )

        collection = Collection(collection_name, using=alias)
        collection.load()

        # 分批查询指定ID的向量以避免Milvus查询限制
        batch_size = 10000  # 设置批次大小，低于16384的限制
        all_results = []

        for i in range(0, len(low_ids), batch_size):
            batch_ids = low_ids[i:i + batch_size]
            expr = f"{id_field} in {batch_ids}"
            batch_results = collection.query(
                expr=expr,
                output_fields=[id_field, vector_field]
            )
            all_results.extend(batch_results)

        # 按ID排序确保顺序一致
        all_results.sort(key=lambda x: x[id_field])

        rec_payloads = {b: [] for b in range(BLOCK_COUNT)}
        device = wm.device

        for result in all_results:
            vec = np.array(result[vector_field], dtype=np.float32)
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

    finally:
        connections.disconnect(alias)


def backup_vectors(db_params, collection_name, id_field, vector_field, low_ids,
                   file_path='original_vectors_backup.npz'):
    """
    备份原始向量
    """
    alias = f"backup_{uuid.uuid4().hex[:8]}"
    try:
        connections.connect(
            alias=alias,
            host=db_params.get("host", "localhost"),
            port=db_params.get("port", 19530)
        )

        collection = Collection(collection_name, using=alias)
        collection.load()

        # 分批查询指定ID的向量以避免Milvus查询限制
        batch_size = 10000  # 设置批次大小，低于16384的限制
        backup_data = {}

        for i in range(0, len(low_ids), batch_size):
            batch_ids = low_ids[i:i + batch_size]
            expr = f"{id_field} in {batch_ids}"
            batch_results = collection.query(
                expr=expr,
                output_fields=[id_field, vector_field]
            )

            for result in batch_results:
                backup_data[str(result[id_field])] = np.array(result[vector_field], dtype=np.float32)

        np.savez(file_path, **backup_data)
        return len(backup_data)

    finally:
        connections.disconnect(alias)


def embed_watermark(db_params, collection_name, id_field, vector_field, message, embed_rate=None, total_vecs=None,
                    ids_file=None):
    """
    嵌入水印流程 - 使用嵌入率选择向量
    
    Args:
        db_params: 数据库连接参数
        collection_name: 集合名
        id_field: 主键字段名
        vector_field: 向量字段名
        message: 水印消息
        embed_rate: 水印嵌入率（0-1之间的浮点数），优先使用此参数
        total_vecs: 兼容性参数，已弃用
        ids_file: ID文件路径（可选）
        
    Returns:
        嵌入结果字典
    """
    # 参数处理：优先使用embed_rate
    if embed_rate is None:
        embed_rate = DEFAULT_EMBED_RATE

    if not (0 < embed_rate <= 1):
        return {"success": False, "error": f"水印嵌入率必须在(0,1]范围内，当前值：{embed_rate}"}

    # 检查消息长度
    if len(message) != BLOCK_COUNT * 2:
        return {"success": False, "error": f"消息长度必须为 {BLOCK_COUNT * 2} 字符"}

    # 1) 分块消息数据
    chunks = partition_message(message)

    # 2) 拉原始集合向量
    try:
        ids, data = fetch_vectors(db_params, collection_name, id_field, vector_field)
    except Exception as e:
        return {"success": False, "error": f"获取向量数据失败: {str(e)}"}

    # 3) 构建索引和生成ID列表
    try:
        idx = build_hnsw_index(data.copy())
        in_deg = compute_in_degrees(idx, ids)

        # 使用嵌入率选择向量
        low_ids = select_low_degree_ids_by_rate(ids, in_deg, embed_rate)

        # 如果选出的向量太少，返回错误
        if len(low_ids) < BLOCK_COUNT:
            return {
                "success": False,
                "error": f"按{embed_rate:.1%}嵌入率选择的向量数量({len(low_ids)})少于最小需求({BLOCK_COUNT})"
            }

        # 保存ID列表到文件（如果提供了文件路径）
        if ids_file:
            save_low_degree_ids(low_ids, ids_file)
    except Exception as e:
        return {"success": False, "error": f"生成低入度向量ID失败: {str(e)}"}

    # 4) 嵌入并写回原始集合
    try:
        wm = VectorWatermark(vec_dim=DIM, msg_len=MSG_LEN, model_path=MODEL_PATH)
        updated = embed_into_milvus(low_ids, data, chunks, wm, db_params, collection_name, id_field, vector_field)

        # 计算实际嵌入率
        actual_rate = len(low_ids) / len(ids) if len(ids) > 0 else 0

        # 添加使用的向量数量到返回信息
        return {
            "success": True,
            "message": f"水印嵌入完成，按{embed_rate:.1%}嵌入率选择了{len(low_ids)}个向量，更新了{updated}个向量",
            "updated": updated,
            "used_vectors": len(low_ids),
            "total_vectors": len(ids),
            "embed_rate": actual_rate,
            "ids_file": ids_file
        }
    except Exception as e:
        return {"success": False, "error": f"嵌入水印失败: {str(e)}"}


def extract_watermark(db_params, collection_name, id_field, vector_field, embed_rate=None, ids_file=None):
    """
    提取水印流程 - 重新计算低入度节点，不依赖ID文件
    
    Args:
        db_params: 数据库连接参数
        collection_name: 集合名
        id_field: 主键字段名
        vector_field: 向量字段名
        embed_rate: 水印嵌入率（0-1之间的浮点数），如果提供则使用此参数重新计算
        ids_file: ID文件路径（可选，用于向后兼容）
        
    Returns:
        提取结果字典
    """
    # 参数处理
    if embed_rate is None:
        embed_rate = DEFAULT_EMBED_RATE

    # 1) 如果提供了ID文件且文件存在，则使用文件中的ID列表
    low_ids = None
    if ids_file and os.path.exists(ids_file):
        try:
            low_ids = load_low_degree_ids(ids_file)
            print(f"使用ID文件中的{len(low_ids)}个向量ID进行提取")
        except Exception as e:
            print(f"加载ID文件失败，将重新计算: {str(e)}")
            low_ids = None

    # 2) 如果没有有效的ID列表，则重新计算
    if low_ids is None:
        try:
            # 获取所有向量数据
            ids, data = fetch_vectors(db_params, collection_name, id_field, vector_field)

            # 构建索引并计算入度
            idx = build_hnsw_index(data.copy())
            in_deg = compute_in_degrees(idx, ids)

            # 使用嵌入率选择向量
            low_ids = select_low_degree_ids_by_rate(ids, in_deg, embed_rate)
            print(f"重新计算得到{len(low_ids)}个低入度向量ID")

        except Exception as e:
            return {"success": False, "error": f"重新计算低入度向量ID失败: {str(e)}"}

    # 3) 提取水印
    try:
        wm = VectorWatermark(vec_dim=DIM, msg_len=MSG_LEN, model_path=MODEL_PATH)
        rec = extract_from_milvus(db_params, collection_name, id_field, vector_field, low_ids, wm)
    except Exception as e:
        return {"success": False, "error": f"提取水印失败: {str(e)}"}

    # 4) 恢复消息 - 使用纯统计方法
    try:
        from collections import Counter
        import numpy as np

        recovered = []
        recovered_blocks = 0
        stats_info = []  # 记录每个区块的统计信息

        for b in range(BLOCK_COUNT):
            samples = rec.get(b, [])
            if not samples:
                recovered.append("??")
                stats_info.append({"block": b, "status": "empty", "samples": 0})
                continue

            recovered_blocks += 1

            # 将每个样本转化为比特串的哈希值用于统计
            # 注意: 我们统计的是原始比特串而不是转换后的文本
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
                # 使用出现频率最高的比特串，无论其频率如何
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
            "method": "pure_statistical",
            "used_vectors": len(low_ids),
            "embed_rate": embed_rate,
            "stats": stats_info  # 添加统计信息以便分析
        }
    except Exception as e:
        return {"success": False, "error": f"恢复消息失败: {str(e)}"}
