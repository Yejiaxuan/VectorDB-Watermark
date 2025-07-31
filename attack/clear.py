import psycopg2
import os

DB_PARAMS = {
    'host': 'localhost',
    'port': 5555,
    'dbname': 'pgvector',
    'user': 'pgvector',
    'password': 'pgvector',
}
TABLE_NAME = 'items'

conn = psycopg2.connect(**DB_PARAMS)
cur = conn.cursor()

cur.execute(f"DELETE FROM {TABLE_NAME};")
conn.commit()

print(f"✅ 已清空表 `{TABLE_NAME}` 的所有向量数据。")

cur.close()
conn.close()
