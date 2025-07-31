import psycopg2
import random

# === 数据库连接参数 ===
DB_HOST = "localhost"
DB_PORT = 5555
DB_NAME = "pgvector"
DB_USER = "pgvector"
DB_PASSWORD = "pgvector"
TABLE_NAME = "items"
PRIMARY_KEY = "id"

# === 子集攻击参数 ===
KEEP_RATIO = 0.9  # 只保留 % 的向量
SEED = 42         # 随机种子，保证实验可复现

random.seed(SEED)

def main():
    # === 连接数据库 ===
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()

    # === 获取全部 ID ===
    print("📥 正在获取全部向量 ID ...")
    cur.execute(f"SELECT {PRIMARY_KEY} FROM {TABLE_NAME}")
    all_ids = [row[0] for row in cur.fetchall()]
    total = len(all_ids)

    # === 随机选择保留子集 ===
    keep_num = int(total * KEEP_RATIO)
    keep_ids = set(random.sample(all_ids, keep_num))
    remove_ids = set(all_ids) - keep_ids
    print(f"🎯 共 {total} 条向量，保留 {keep_num} 条，删除 {len(remove_ids)} 条")

    # === 执行删除操作 ===
    if remove_ids:
        print("🧨 开始执行删除 ...")
        cur.execute(
            f"DELETE FROM {TABLE_NAME} WHERE {PRIMARY_KEY} NOT IN %s",
            (tuple(keep_ids),)
        )
        conn.commit()
        print("✅ 子集攻击完成")
    else:
        print("⚠️ 无需删除，全部保留")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
