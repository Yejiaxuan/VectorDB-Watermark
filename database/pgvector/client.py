"""
PGVector管理器类

封装所有与 pgvector 相关的数据库操作，包括：
- 数据库连接管理
- 表和列的查询
- 主键查询
- 水印嵌入和提取
- 文件管理
"""

import os
import uuid
import psycopg2
from datetime import datetime
from typing import Dict, Any, Optional
from pgvector.psycopg2 import register_vector
from .pg_func import embed_watermark, extract_watermark


class PGVectorManager:
    """PGVector数据库操作管理器"""

    def __init__(self):
        """
        初始化管理器
        """
        pass

    def test_connection(self, db_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        测试数据库连接
        
        Args:
            db_params: 数据库连接参数
            
        Returns:
            连接结果字典
        """
        try:
            conn = psycopg2.connect(**db_params)
            register_vector(conn)
            conn.close()
            return {"success": True, "message": "连接成功"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_tables(self, db_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        列出 public schema 下所有表名
        
        Args:
            db_params: 数据库连接参数
            
        Returns:
            包含表名列表的字典
        """
        try:
            conn = psycopg2.connect(**db_params)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT table_name
                  FROM information_schema.tables
                 WHERE table_schema = 'public'
                   AND table_type = 'BASE TABLE';
                """
            )
            tables = [r[0] for r in cur.fetchall()]
            cur.close()
            conn.close()
            return {"success": True, "tables": tables}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_vector_columns(self, db_params: Dict[str, Any], table: str) -> Dict[str, Any]:
        """
        列出指定表中所有 pgvector 类型的列
        
        Args:
            db_params: 数据库连接参数
            table: 表名
            
        Returns:
            包含向量列名列表的字典
        """
        try:
            conn = psycopg2.connect(**db_params)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = 'public'
                   AND table_name = %s
                   AND udt_name = 'vector';
                """,
                (table,)
            )
            cols = [r[0] for r in cur.fetchall()]
            cur.close()
            conn.close()
            return {"success": True, "columns": cols}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_primary_keys(self, db_params: Dict[str, Any], table: str) -> Dict[str, Any]:
        """
        列出指定表的主键列
        
        Args:
            db_params: 数据库连接参数
            table: 表名
            
        Returns:
            包含主键列名列表的字典
        """
        try:
            conn = psycopg2.connect(**db_params)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT kc.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kc
                    ON kc.constraint_name = tc.constraint_name
                WHERE
                    tc.constraint_type = 'PRIMARY KEY' AND
                    tc.table_schema = 'public' AND
                    tc.table_name = %s
                ORDER BY kc.ordinal_position;
                """,
                (table,)
            )
            keys = [r[0] for r in cur.fetchall()]
            cur.close()
            conn.close()
            return {"success": True, "keys": keys}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def embed_watermark(
            self,
            db_params: Dict[str, Any],
            table: str,
            id_column: str,
            vector_column: str,
            message: str,
            embed_rate: float = 0.1,
            total_vecs: int = 1600
    ) -> Dict[str, Any]:
        """
        在指定表的向量列中嵌入水印，不生成ID文件
        
        Args:
            db_params: 数据库连接参数
            table: 表名
            id_column: 主键列名
            vector_column: 向量列名
            message: 水印消息
            embed_rate: 水印嵌入率（0-1之间的浮点数），默认10%
            total_vecs: 使用的向量数量，默认1600（已弃用，保留兼容性）
            
        Returns:
            嵌入结果字典
        """
        try:
            # 调用水印嵌入函数
            result = embed_watermark(
                db_params=db_params,
                table_name=table,
                id_col=id_column,
                emb_col=vector_column,
                message=message,
                embed_rate=embed_rate
            )

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_watermark(
            self,
            db_params: Dict[str, Any],
            table: str,
            id_column: str,
            vector_column: str,
            embed_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        从指定表的向量列中提取水印，重新计算低入度节点
        
        Args:
            db_params: 数据库连接参数
            table: 表名
            id_column: 主键列名
            vector_column: 向量列名
            embed_rate: 水印嵌入率（0-1之间的浮点数），默认10%
            
        Returns:
            提取结果字典
        """
        try:
            # 调用水印提取函数
            result = extract_watermark(
                db_params=db_params,
                table_name=table,
                id_col=id_column,
                emb_col=vector_column,
                embed_rate=embed_rate
            )

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
