import os
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
import numpy as np

# —— 配置区 ——
DB_PARAMS = {
    'host':     os.getenv('PG_HOST', 'localhost'),
    'port':     int(os.getenv('PG_PORT', 5555)),
    'dbname':   os.getenv('PG_DATABASE', 'pgvector'),
    'user':     os.getenv('PG_USER', 'pgvector'),
    'password': os.getenv('PG_PASSWORD', 'pgvector'),
}
TABLE_NAME = os.getenv('TABLE_NAME', 'items')
VECTOR_COLUMN = os.getenv('VECTOR_COLUMN', 'emb')
NPY_PATH = os.getenv('NPY_PATH', 'nq_qa_combined_384d.npy')
VECTOR_DIM = 384  # 修改这里即可支持不同维度
# ——————————

def validate_sql_identifier(name: str):
    """确保表名和列名合法，防止 SQL 注入"""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"非法的 SQL 标识符: {name}")
    return name

def main():
    # 验证 SQL 标识符
    validate_sql_identifier(TABLE_NAME)
    validate_sql_identifier(VECTOR_COLUMN)

    # 加载 .npy 向量文件
    if not os.path.isfile(NPY_PATH):
        raise FileNotFoundError(f"❌ 向量文件未找到: {NPY_PATH}")

    data = np.load(NPY_PATH)

    if data.ndim != 2 or data.shape[1] != VECTOR_DIM:
        raise ValueError(f"❌ 期望形状为 (N, {VECTOR_DIM})，但实际为: {data.shape}")

    print(f"📥 加载成功，共 {data.shape[0]} 条向量，维度为 {data.shape[1]}")

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        register_vector(conn)
        cur = conn.cursor()

        # 自动创建 pgvector 扩展
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 自动建表（如不存在）
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                {VECTOR_COLUMN} VECTOR({VECTOR_DIM})
            );
        """)
        conn.commit()

        # 批量准备数据
        values = [(vec.tolist(),) for vec in data.astype(np.float32)]

        print(f"📤 正在插入向量到表 `{TABLE_NAME}` 的列 `{VECTOR_COLUMN}` ...")
        sql = f"INSERT INTO {TABLE_NAME} ({VECTOR_COLUMN}) VALUES %s;"
        execute_values(cur, sql, values, template="(%s)")
        conn.commit()

        print(f"✅ 成功插入 {len(values)} 条向量。")

    except Exception as e:
        print(f"❌ 插入失败: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    main()
