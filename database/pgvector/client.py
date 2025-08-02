"""
PGVector管理器类

封装所有与 pgvector 相关的数据库操作，包括：
- 数据库连接管理
- 表和列的查询
- 主键查询
- 水印嵌入和提取
- 向量维度获取
- 训练数据获取
"""

import os
import uuid
import psycopg2
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pgvector.psycopg2 import register_vector
from .pg_func import embed_watermark, extract_watermark, reduce_dimensions


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

    def get_vector_dimension(self, db_params: Dict[str, Any], table: str, vector_column: str) -> Dict[str, Any]:
        """
        获取指定向量列的维度
        
        Args:
            db_params: 数据库连接参数
            table: 表名
            vector_column: 向量列名
            
        Returns:
            包含向量维度的字典
        """
        try:
            conn = psycopg2.connect(**db_params)
            register_vector(conn)
            cur = conn.cursor()
            
            # 获取一个向量样本来确定维度
            cur.execute(f"SELECT {vector_column} FROM {table} WHERE {vector_column} IS NOT NULL LIMIT 1")
            result = cur.fetchone()
            
            if result and result[0] is not None:
                vector_dim = len(result[0])
                cur.close()
                conn.close()
                return {"success": True, "dimension": vector_dim}
            else:
                cur.close()
                conn.close()
                return {"success": False, "error": "表中没有有效的向量数据"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_training_data(self, db_params: Dict[str, Any], table: str, vector_column: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        从数据库获取训练数据
        
        Args:
            db_params: 数据库连接参数
            table: 表名
            vector_column: 向量列名
            limit: 限制返回的向量数量，None表示获取所有数据
            
        Returns:
            包含向量数据的字典
        """
        try:
            conn = psycopg2.connect(**db_params)
            register_vector(conn)
            cur = conn.cursor()
            
            # 构建查询语句
            query = f"SELECT {vector_column} FROM {table} WHERE {vector_column} IS NOT NULL"
            if limit is not None:
                query += f" LIMIT {limit}"
            
            cur.execute(query)
            results = cur.fetchall()
            
            if results:
                # 将向量数据转换为numpy数组
                vectors = np.array([list(row[0]) for row in results])
                cur.close()
                conn.close()
                return {"success": True, "vectors": vectors, "count": len(vectors)}
            else:
                cur.close()
                conn.close()
                return {"success": False, "error": "表中没有有效的向量数据"}
                
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
            encryption_key: str = None,
            total_vecs: int = 1600
    ) -> Dict[str, Any]:
        """
        在指定表的向量列中嵌入水印，使用AES-GCM加密明文消息，支持伪随机载体选择
        
        Args:
            db_params: 数据库连接参数
            table: 表名
            id_column: 主键列名
            vector_column: 向量列名
            message: 明文消息（16字节）
            embed_rate: 水印嵌入率（0-1之间的浮点数），默认10%
            encryption_key: AES-GCM加密密钥，同时用作伪随机载体选择的种子
            total_vecs: 使用的向量数量，默认1600（已弃用，保留兼容性）
            
        Returns:
            嵌入结果字典，包含nonce供用户保存
        """
        try:
            # 验证明文消息长度
            if len(message) != 16:
                return {"success": False, "error": "明文消息长度必须为16字节"}
            
            # 验证加密密钥
            if not encryption_key:
                return {"success": False, "error": "必须提供AES-GCM加密密钥"}
            
            # 首先获取向量维度
            dim_result = self.get_vector_dimension(db_params, table, vector_column)
            if not dim_result["success"]:
                return dim_result
            
            vec_dim = dim_result["dimension"]
            
            # 调用水印嵌入函数
            result = embed_watermark(
                db_params=db_params,
                table_name=table,
                id_col=id_column,
                emb_col=vector_column,
                message=message,
                vec_dim=vec_dim,
                embed_rate=embed_rate,
                encryption_key=encryption_key
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
            embed_rate: float = 0.1,
            encryption_key: str = None,
            nonce: str = None
    ) -> Dict[str, Any]:
        """
        从指定表的向量列中提取水印，使用AES-GCM解密得到明文消息，支持伪随机载体选择
        
        Args:
            db_params: 数据库连接参数
            table: 表名
            id_column: 主键列名
            vector_column: 向量列名
            embed_rate: 水印嵌入率（0-1之间的浮点数），默认10%
            encryption_key: AES-GCM解密密钥，同时用作伪随机载体选择的种子
            nonce: nonce的十六进制表示，必须提供用于解密
            
        Returns:
            提取结果字典
        """
        try:
            # 验证解密密钥
            if not encryption_key:
                return {"success": False, "error": "必须提供AES-GCM解密密钥"}
            
            # 验证nonce
            if not nonce:
                return {"success": False, "error": "必须提供nonce用于解密"}
            
            # 首先获取向量维度
            dim_result = self.get_vector_dimension(db_params, table, vector_column)
            if not dim_result["success"]:
                return dim_result
            
            vec_dim = dim_result["dimension"]
            
            # 调用水印提取函数
            result = extract_watermark(
                db_params=db_params,
                table_name=table,
                id_col=id_column,
                emb_col=vector_column,
                vec_dim=vec_dim,
                embed_rate=embed_rate,
                encryption_key=encryption_key,
                nonce_hex=nonce
            )

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
        
    def get_embedding_visualization(self, original_vectors, embedded_vectors, method="tsne", n_samples=500):
        """
        获取嵌入前后向量的降维可视化数据
        
        Args:
            original_vectors: 原始向量数组
            embedded_vectors: 嵌入水印后的向量数组
            method: 降维方法，可选 "tsne" 或 "pca"
            n_samples: 用于降维的最大样本数，None表示不限制
            
        Returns:
            降维后的可视化数据
        """
        try:
            result = reduce_dimensions(
                np.array(original_vectors), 
                np.array(embedded_vectors),
                method=method,
                n_samples=n_samples
            )
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}
