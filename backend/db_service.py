import psycopg2
from pgvector.psycopg2 import register_vector
from .models import DBParams

def test_connect(params: DBParams):
    conn = psycopg2.connect(**params.dict())
    register_vector(conn)
    conn.close()
