"""
Milvus向量水印系统核心函数库 - 支持AES-GCM加密

提供向量水印嵌入和提取的关键功能：
- 支持参数化Milvus连接和集合结构
- 支持自定义主键字段名和向量字段名
- 集成AES-GCM加密算法处理明文消息
- 通过文件保存低入度ID列表确保嵌入/提取一致性
"""
import os
import json
import uuid
import hashlib
import base64
import numpy as np
import faiss
import torch
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from algorithms.deep_learning.watermark import VectorWatermark
from configs.config import Config
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

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


def fetch_vectors(db_params, collection_name, id_field, vector_field, vec_dim):
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
            return [], np.array([]).reshape(0, vec_dim)

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
        data = np.vstack(vectors) if vectors else np.array([]).reshape(0, vec_dim)

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


def build_hnsw_index(data: np.ndarray, vec_dim: int):
    """
    构建HNSW索引（在嵌入水印时使用）
    """
    idx = faiss.IndexHNSWFlat(vec_dim, M)
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

    stegos = []
    original_vectors = []  # 新增：保存原始向量
    embedded_vectors = []  # 新增：保存嵌入后的向量
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
        original_vectors.append(vec.copy())  # 保存原始向量副本
        
        # 记录原始向量的范数
        original_norm = np.linalg.norm(vec)
        
        # 归一化向量
        normalized_vec = vec / (original_norm + 1e-8)
        cover = torch.tensor(normalized_vec, device=device).unsqueeze(0)

        msg_t = torch.tensor([msg_bits], dtype=torch.float32, device=device)
        stego, _ = wm.encode(cover, message=msg_t)
        
        # 将结果转换回numpy，并恢复原始范数
        stego_np = stego.squeeze(0).cpu().numpy()
        stego_restored = stego_np * original_norm
        stegos.append(stego_restored.astype(np.float32))
        embedded_vectors.append(stego_restored.astype(np.float32))  # 保存嵌入后的向量

    # 打印每个区块分配的向量数量
    print(f"Block distribution: {block_counters}")

    # 更新向量到Milvus
    updated = update_vectors(db_params, collection_name, id_field, vector_field, low_ids, stegos)
    
    # 修改：返回更新数量和嵌入前后的向量
    return updated, np.array(original_vectors), np.array(embedded_vectors)


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


# —— AES-GCM 加密解密函数 —— #

def derive_key_from_password(password: str, salt: bytes = None) -> bytes:
    """
    从密码派生AES密钥
    """
    if salt is None:
        # 使用固定盐值确保同一密码生成相同密钥
        salt = b'DbWM_Salt_2024'
    
    # 使用PBKDF2派生32字节密钥
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return key


def aes_gcm_encrypt(plaintext: str, password: str) -> tuple:
    """
    使用AES-GCM加密明文消息，生成随机nonce
    
    Args:
        plaintext: 16字节明文消息
        password: 用户提供的密码
        
    Returns:
        tuple: (encrypted_data, nonce)
            - encrypted_data: 24字节数据（16字节密文 + 8字节认证标签）
            - nonce: 12字节随机nonce，需要用户保存用于后续解密
    """
    if len(plaintext) != 16:
        raise ValueError("明文消息必须为16字节")
    
    # 派生密钥
    key = derive_key_from_password(password)
    
    # 生成随机nonce
    nonce = get_random_bytes(12)
    
    # AES-GCM加密，指定8字节的标签长度
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=8)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))
    
    # 返回密文+标签和nonce
    return ciphertext + tag, nonce


def aes_gcm_decrypt(encrypted_data: bytes, password: str, nonce: bytes) -> str:
    """
    使用AES-GCM解密密文数据，需要提供正确的nonce
    
    Args:
        encrypted_data: 24字节加密数据（16字节密文 + 8字节标签）
        password: 用户提供的密码
        nonce: 12字节nonce，必须与加密时使用的相同
        
    Returns:
        16字节明文消息
    """
    if len(encrypted_data) != 24:
        raise ValueError("加密数据必须为24字节")
    
    if len(nonce) != 12:
        raise ValueError("nonce必须为12字节")
    
    # 分离密文和标签
    ciphertext = encrypted_data[:16]
    tag = encrypted_data[16:]
    
    # 派生密钥
    key = derive_key_from_password(password)
    
    # AES-GCM解密，指定8字节的标签长度
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=8)
    try:
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext.decode('utf-8')
    except ValueError as e:
        raise ValueError(f"解密失败：密钥错误或数据损坏 - {str(e)}")


def encrypt_message_to_32bytes(plaintext: str, password: str) -> tuple:
    """
    将16字节明文加密为恰好32字符的Base64字符串，用于水印嵌入
    
    Args:
        plaintext: 16字节明文消息
        password: 加密密钥
        
    Returns:
        tuple: (encrypted_str, nonce_hex)
            - encrypted_str: 32字符Base64字符串
            - nonce_hex: nonce的十六进制表示，用户需保存以便后续解密
    """
    encrypted_bytes, nonce = aes_gcm_encrypt(plaintext, password)
    
    # 使用Base64编码 - 24字节数据会精确编码为32个字符
    encrypted_str = base64.b64encode(encrypted_bytes).decode('ascii')
    
    # 检查编码结果是否符合预期
    if len(encrypted_str) != 32:
        print(f"警告: Base64编码结果长度为 {len(encrypted_str)}，预期32")
    
    # 将nonce转换为十六进制字符串方便用户保存
    nonce_hex = nonce.hex()
    return encrypted_str, nonce_hex


def decrypt_32bytes_to_message(encrypted_str: str, password: str, nonce_hex: str) -> str:
    """
    将32字符Base64字符串解密为16字节明文消息，需要提供正确的nonce
    
    Args:
        encrypted_str: 32字符Base64字符串
        password: 解密密钥
        nonce_hex: nonce的十六进制表示
        
    Returns:
        16字节明文消息
    """
    if len(encrypted_str) != 32:
        raise ValueError("加密字符串长度必须为32")
    
    # 将nonce的十六进制表示转换回字节
    try:
        nonce_hex = nonce_hex.strip()
        print(f"DEBUG - 输入的nonce (hex): '{nonce_hex}'")
        nonce = bytes.fromhex(nonce_hex)
        print(f"DEBUG - 解析的nonce字节: {nonce.hex()}")
    except ValueError as e:
        print(f"DEBUG - nonce解析错误: {str(e)}")
        raise ValueError(f"无效的nonce十六进制表示: {str(e)}")
    
    # 直接Base64解码
    try:
        print(f"DEBUG - 输入的加密字符串: '{encrypted_str}'")
        encrypted_bytes = base64.b64decode(encrypted_str)
        print(f"DEBUG - 转换后的字节 (hex): {encrypted_bytes.hex()}")
        print(f"DEBUG - 转换后的字节长度: {len(encrypted_bytes)} 字节")
        
        # 派生密钥用于调试
        key = derive_key_from_password(password)
        print(f"DEBUG - 派生的密钥 (hex): {key.hex()}")
        
        # 检查长度
        if len(encrypted_bytes) != 24:
            print(f"WARNING - 解码后的密文+标签长度不是24字节! 实际: {len(encrypted_bytes)}")
        
        return aes_gcm_decrypt(encrypted_bytes, password, nonce)
    except Exception as e:
        print(f"DEBUG - 转换或解密过程异常: {str(e)}")
        raise e


def embed_watermark(db_params, collection_name, id_field, vector_field, message, vec_dim, embed_rate=None, encryption_key=None, total_vecs=None, ids_file=None):
    """
    嵌入水印流程 - 支持AES-GCM加密的明文消息
    
    Args:
        db_params: 数据库连接参数
        collection_name: 集合名
        id_field: 主键字段名
        vector_field: 向量字段名
        message: 明文消息（16字节）
        vec_dim: 向量维度
        embed_rate: 水印嵌入率（0-1之间的浮点数），优先使用此参数
        encryption_key: AES-GCM加密密钥
        total_vecs: 使用的向量数量（已弃用，保留兼容性）
        ids_file: ID文件路径（已弃用，保留兼容性）
    """
    # 验证明文消息长度
    if len(message) != 16:
        return {"success": False, "error": f"明文消息长度必须为16字符，当前为{len(message)}字符"}
    
    # 验证加密密钥
    if not encryption_key:
        return {"success": False, "error": "必须提供AES-GCM加密密钥"}

    # 确定嵌入率
    if embed_rate is None:
        embed_rate = DEFAULT_EMBED_RATE

    if not (0 < embed_rate <= 1):
        return {"success": False, "error": "水印嵌入率必须在0和1之间"}

    try:
        # 1) 使用AES-GCM加密明文消息为32字节，并获取nonce
        encrypted_message, nonce_hex = encrypt_message_to_32bytes(message, encryption_key)
        print(f"明文消息: {message}")
        print(f"加密后消息: {encrypted_message}")
        print(f"生成的nonce: {nonce_hex}")
        
        # 2) 分块加密消息数据
        chunks = partition_message(encrypted_message)

        # 3) 拉原始集合向量
        ids, data = fetch_vectors(db_params, collection_name, id_field, vector_field, vec_dim)
    except Exception as e:
        return {"success": False, "error": f"数据准备失败: {str(e)}"}

    # 4) 构建索引和生成ID列表
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

    # 5) 嵌入并写回原始集合
    try:
        model_path = Config.get_model_path(vec_dim)
        wm = VectorWatermark(vec_dim=vec_dim, msg_len=MSG_LEN, model_path=model_path)
        
        # 修改：接收嵌入前后的向量
        updated, orig_samples, emb_samples = embed_into_milvus(low_ids, data, chunks, wm, db_params, collection_name, id_field, vector_field)
        
        # 计算实际嵌入率
        actual_rate = len(low_ids) / len(ids)
        
        # 直接使用返回的向量进行降维和可视化
        visualization_data = reduce_dimensions(
            orig_samples,  # 直接使用嵌入前的向量
            emb_samples    # 直接使用嵌入后的向量
        )

        # 返回嵌入结果
        return {
            "success": True,
            "message": f"明文消息加密后嵌入完成，嵌入率{actual_rate:.1%}({len(low_ids)}/{len(ids)})，更新了{updated}个向量",
            "updated": updated,
            "used_vectors": len(low_ids),
            "total_vectors": len(ids),
            "embed_rate": actual_rate,
            "plaintext": message,
            "encrypted_preview": encrypted_message[:8] + "...",  # 只显示前8字符
            "nonce": nonce_hex,  # 返回nonce供用户保存
            "visualization_data": visualization_data  # 添加可视化数据
        }
    except Exception as e:
        return {"success": False, "error": f"嵌入水印失败: {str(e)}"}


def extract_watermark(db_params, collection_name, id_field, vector_field, vec_dim, embed_rate=None, encryption_key=None, nonce_hex=None, ids_file=None):
    """
    提取水印流程 - 支持AES-GCM解密得到明文消息
    
    Args:
        db_params: 数据库连接参数
        collection_name: 集合名
        id_field: 主键字段名
        vector_field: 向量字段名
        vec_dim: 向量维度
        embed_rate: 水印嵌入率（0-1之间的浮点数），如果提供则使用此参数重新计算
        encryption_key: AES-GCM解密密钥
        nonce_hex: nonce的十六进制表示，必须提供用于解密
        ids_file: ID文件路径（已弃用，保留兼容性）
    """
    # 验证解密密钥和nonce
    if not encryption_key:
        return {"success": False, "error": "必须提供AES-GCM解密密钥"}
    
    if not nonce_hex:
        return {"success": False, "error": "必须提供nonce用于解密"}
        
    try:
        # 验证nonce格式
        bytes.fromhex(nonce_hex)
    except ValueError:
        return {"success": False, "error": "无效的nonce格式，应为十六进制字符串"}

    # 确定嵌入率
    if embed_rate is None:
        embed_rate = DEFAULT_EMBED_RATE

    if not (0 < embed_rate <= 1):
        return {"success": False, "error": "水印嵌入率必须在0和1之间"}

    # 1) 重新获取向量数据并计算低入度节点
    try:
        ids, data = fetch_vectors(db_params, collection_name, id_field, vector_field, vec_dim)

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

        rec_payloads = extract_from_milvus(db_params, collection_name, id_field, vector_field, low_ids, wm)

        # 统计有效解码
        valid_decodes = sum(len(samples) for samples in rec_payloads.values())
        total_decodes = len(low_ids)

        print(f"总解码: {total_decodes}, 有效解码: {valid_decodes}")

        # 3) 恢复加密消息 - 使用纯统计方法
        from collections import Counter

        recovered = []
        recovered_blocks = 0
        stats_info = []

        for b in range(BLOCK_COUNT):
            samples = rec_payloads.get(b, [])
            if not samples:
                recovered.append("?")
                stats_info.append({"block": b, "status": "empty", "samples": 0})
                continue

            recovered_blocks += 1

            # 将每个样本转化为比特串的哈希值用于统计
            bit_strings = []
            for sample in samples:
                bits = (sample > 0.5).astype(int)
                bit_tuple = tuple(bits.tolist())
                bit_strings.append(bit_tuple)

            # 统计比特串的出现频率
            counter = Counter(bit_strings)
            most_common = counter.most_common(1)

            if most_common:
                most_common_bits, count = most_common[0]
                best_bits = np.array(most_common_bits)
                char = bits_to_text(best_bits)

                stats_info.append({
                    "block": b,
                    "status": "found",
                    "samples": len(samples),
                    "most_common_count": count,
                    "most_common_percent": round(count / len(samples) * 100, 2)
                })
            else:
                char = "?"
                stats_info.append({"block": b, "status": "no_results", "samples": len(samples)})

            recovered.append(char)

        encrypted_message = "".join(recovered)
        print(f"恢复的加密消息: {encrypted_message}")

        # 4) 使用AES-GCM解密得到明文消息
        try:
            if len(encrypted_message) == 32 and "?" not in encrypted_message:
                plaintext = decrypt_32bytes_to_message(encrypted_message, encryption_key, nonce_hex)
                print(f"解密得到明文: {plaintext}")
                
                return {
                    "success": True,
                    "message": plaintext,
                    "blocks": BLOCK_COUNT,
                    "recovered": recovered_blocks,
                    "method": "recomputed_statistical_with_aes_gcm",
                    "total_low_degree_nodes": len(low_ids),
                    "valid_decodes": valid_decodes,
                    "total_decodes": total_decodes,
                    "encrypted_message": encrypted_message,
                    "stats": stats_info
                }
            else:
                # 如果加密消息不完整，仍然返回但标注解密失败
                return {
                    "success": False,
                    "error": f"加密消息不完整或损坏，无法解密。恢复的消息: {encrypted_message}",
                    "blocks": BLOCK_COUNT,
                    "recovered": recovered_blocks,
                    "encrypted_message": encrypted_message,
                    "stats": stats_info
                }
        except Exception as decrypt_error:
            return {
                "success": False,
                "error": f"解密失败: {str(decrypt_error)}。恢复的加密消息: {encrypted_message}",
                "blocks": BLOCK_COUNT,
                "recovered": recovered_blocks,
                "encrypted_message": encrypted_message,
                "stats": stats_info
            }

    except Exception as e:
        return {"success": False, "error": f"提取水印失败: {str(e)}"}


def reduce_dimensions(original_vectors, embedded_vectors, method="tsne", n_samples=None):
    """优化后的降维算法实现"""
    # 使用预处理 + 多步降维策略
    orig_samples = original_vectors
    emb_samples = embedded_vectors
    
    # 合并向量以便一起降维
    combined = np.vstack([orig_samples, emb_samples])
    
    if method == "tsne":
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # 预处理步骤: 先用PCA将高维向量降到中间维度(如50维)
        if combined.shape[1] > 50:
            pca = PCA(n_components=50)
            combined_reduced = pca.fit_transform(combined)
        else:
            combined_reduced = combined
            
        # 使用优化的t-SNE参数
        tsne = TSNE(
            n_components=2, 
            perplexity=min(30, combined_reduced.shape[0] // 5),  # 动态调整perplexity
            n_iter=1000,
            method='barnes_hut',  # 使用更快的近似算法
            n_jobs=-1,  # 使用所有CPU核心
            random_state=42
        )
        
        reduced = tsne.fit_transform(combined_reduced)
    else:
        # PCA通常足够快，只需优化参数
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(combined)
    
    # 分离结果
    n_orig = len(orig_samples)
    original_reduced = reduced[:n_orig].tolist()
    embedded_reduced = reduced[n_orig:].tolist()
    
    # 计算欧氏距离
    distances = np.sqrt(np.sum((orig_samples - emb_samples)**2, axis=1))
    avg_distance = float(np.mean(distances))
    max_distance = float(np.max(distances))
    
    # 使用PyTorch的F.cosine_similarity计算余弦相似度，与test.py保持一致
    import torch
    import torch.nn.functional as F
    
    # 转换为PyTorch张量
    orig_tensor = torch.tensor(orig_samples, dtype=torch.float32)
    emb_tensor = torch.tensor(emb_samples, dtype=torch.float32)
    
    # 使用F.cosine_similarity计算余弦相似度
    cosine_similarities = F.cosine_similarity(orig_tensor, emb_tensor, dim=1)
    avg_cosine_similarity = float(cosine_similarities.mean().item())
    
    # 还可以添加最小、最大和标准差，便于更全面的评估
    min_cos_sim = float(cosine_similarities.min().item())
    max_cos_sim = float(cosine_similarities.max().item())
    std_cos_sim = float(cosine_similarities.std().item())
    
    return {
        "original": original_reduced,
        "embedded": embedded_reduced,
        "avg_distance": avg_distance,
        "max_distance": max_distance,
        "avg_cosine_similarity": avg_cosine_similarity,
        "min_cosine_similarity": min_cos_sim,
        "max_cosine_similarity": max_cos_sim,
        "std_cosine_similarity": std_cos_sim,
        "method": method,
        "n_samples": len(original_reduced)
    }