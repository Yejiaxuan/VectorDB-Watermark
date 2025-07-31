import psycopg2
import numpy as np
import ast

# === 数据库连接参数 ===
DB_HOST = "localhost"
DB_PORT = 5555
DB_NAME = "pgvector"
DB_USER = "pgvector"
DB_PASSWORD = "pgvector"
TABLE_NAME = "items"
VECTOR_COLUMN = "emb"
PRIMARY_KEY = "id"

# === 要查看的向量 ID ===
TARGET_ID = 601387  # 可以改成你想看的ID，或改成 LIMIT 1 看第一条

def main():
    # 连接数据库
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()

    print("📥 正在读取指定向量...")

    # 获取指定 ID 的向量
    cur.execute(f"SELECT {PRIMARY_KEY}, {VECTOR_COLUMN} FROM {TABLE_NAME} WHERE {PRIMARY_KEY} = %s", (TARGET_ID,))
    result = cur.fetchone()

    if result is None:
        print(f"❌ 未找到 ID 为 {TARGET_ID} 的向量。")
    else:
        vec_id, vector = result
        vector = np.array(ast.literal_eval(vector)).astype(float)
        print(f"✅ ID: {vec_id}")
        print(f"📏 向量维度: {vector.shape[0]}")
        print(f"🔍 向量内容:\n{vector}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
