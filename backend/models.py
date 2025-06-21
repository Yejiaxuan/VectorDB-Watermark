# backend/models.py

from pydantic import BaseModel

class DBParams(BaseModel):
    host: str
    port: int
    dbname: str
    user: str
    password: str
