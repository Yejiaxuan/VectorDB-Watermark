"""
pgvector_manager.py

PgVector数据库管理类，包含连接、插入、删除、获取、建立索引等所有操作功能。
"""
import os
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values, RealDictCursor
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PgVectorClient:
    """
    PgVector数据库管理类
    提供向量数据库的完整操作功能：连接、插入、删除、查询、索引等
    """
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 dbname: str = None,
                 user: str = None,
                 password: str = None,
                 table_name: str = 'items',
                 vector_column: str = 'emb',
                 vector_dim: int = 384):
        """
        初始化PgVector管理器
        
        Args:
            host: 数据库主机地址
            port: 数据库端口
            dbname: 数据库名
            user: 用户名
            password: 密码
            table_name: 表名
            vector_column: 向量列名
            vector_dim: 向量维度
        """
        # 数据库连接参数
        self.db_params = {
            'host': host or os.getenv('PG_HOST', 'localhost'),
            'port': port or int(os.getenv('PG_PORT', 5432)),
            'dbname': dbname or os.getenv('PG_DATABASE', 'test'),
            'user': user or os.getenv('PG_USER', 'postgres'),
            'password': password or os.getenv('PG_PASSWORD', 'ysj'),
            'table_name': table_name or 'items',
            'vector_column': vector_column or 'emb',
            'vector_dim': vector_dim or 384
        }
        
        # 表和列配置
        self.table_name = table_name
        self.vector_column = vector_column
        self.vector_dim = vector_dim
        
        # 连接对象
        self.conn = None
        self.cursor = None
        
        # FAISS索引缓存
        self.faiss_index = None
        self.id_mapping = None
        
    def connect(self) -> bool:
        """
        连接到数据库
        
        Returns:
            bool: 连接是否成功
        """
        try:
            self.conn = psycopg2.connect(**self.db_params)
            register_vector(self.conn)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("数据库连接已断开")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
    
    def create_table(self, drop_if_exists: bool = False) -> bool:
        """
        创建向量表
        
        Args:
            drop_if_exists: 如果表存在是否删除重建
            
        Returns:
            bool: 创建是否成功
        """
        try:
            if drop_if_exists:
                self.cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                {self.vector_column} VECTOR({self.vector_dim})
            )
            """
            self.cursor.execute(create_sql)
            self.conn.commit()
            logger.info(f"表 {self.table_name} 创建成功")
            return True
        except Exception as e:
            logger.error(f"创建表失败: {e}")
            self.conn.rollback()
            return False
    
    def insert_vector(self, vector: Union[List, np.ndarray], commit: bool = True) -> Optional[int]:
        """
        插入单个向量
        
        Args:
            vector: 向量数据
            commit: 是否立即提交
            
        Returns:
            int: 插入记录的ID，失败返回None
        """
        try:
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            
            insert_sql = f"""
            INSERT INTO {self.table_name} ({self.vector_column}) 
            VALUES (%s) RETURNING id
            """
            self.cursor.execute(insert_sql, (vector,))
            record_id = self.cursor.fetchone()['id']
            
            if commit:
                self.conn.commit()
            
            logger.info(f"向量插入成功，ID: {record_id}")
            return record_id
        except Exception as e:
            logger.error(f"插入向量失败: {e}")
            self.conn.rollback()
            return None
    
    def insert_vectors_batch(self, vectors: Union[List, np.ndarray], 
                           batch_size: int = 1000, commit: bool = True) -> bool:
        """
        批量插入向量
        
        Args:
            vectors: 向量数组
            batch_size: 批次大小
            commit: 是否立即提交
            
        Returns:
            bool: 插入是否成功
        """
        try:
            if isinstance(vectors, np.ndarray):
                vectors = vectors.astype(np.float32)
                values = [(vec.tolist(),) for vec in vectors]
            else:
                values = [(vec,) for vec in vectors]
            
            # 分批插入
            total_inserted = 0
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                insert_sql = f"INSERT INTO {self.table_name} ({self.vector_column}) VALUES %s"
                execute_values(self.cursor, insert_sql, batch, template="(%s)")
                total_inserted += len(batch)
                logger.info(f"已插入 {total_inserted}/{len(values)} 条向量")
            
            if commit:
                self.conn.commit()
            
            logger.info(f"批量插入完成，共插入 {len(values)} 条向量")
            return True
        except Exception as e:
            logger.error(f"批量插入向量失败: {e}")
            self.conn.rollback()
            return False
    
    def insert_vectors_from_npy(self, npy_path: str, batch_size: int = 1000) -> bool:
        """
        从npy文件批量插入向量
        
        Args:
            npy_path: npy文件路径
            batch_size: 批次大小
            
        Returns:
            bool: 插入是否成功
        """
        try:
            if not os.path.isfile(npy_path):
                raise FileNotFoundError(f"向量文件未找到: {npy_path}")
            
            data = np.load(npy_path)
            if data.ndim != 2 or data.shape[1] != self.vector_dim:
                raise ValueError(f"期望形状 (N,{self.vector_dim})，但得到 {data.shape}")
            
            return self.insert_vectors_batch(data, batch_size)
        except Exception as e:
            logger.error(f"从npy文件插入向量失败: {e}")
            return False
    
    def insert_random_vectors(self, num_vectors: int = 5000, batch_size: int = 1000) -> bool:
        """
        插入随机向量（用于测试）
        
        Args:
            num_vectors: 向量数量
            batch_size: 批次大小
            
        Returns:
            bool: 插入是否成功
        """
        try:
            data = np.random.random((num_vectors, self.vector_dim)).astype(np.float32)
            return self.insert_vectors_batch(data, batch_size)
        except Exception as e:
            logger.error(f"插入随机向量失败: {e}")
            return False
    
    def get_vector_by_id(self, vector_id: int) -> Optional[np.ndarray]:
        """
        根据ID获取向量
        
        Args:
            vector_id: 向量ID
            
        Returns:
            np.ndarray: 向量数据，失败返回None
        """
        try:
            select_sql = f"SELECT {self.vector_column} FROM {self.table_name} WHERE id = %s"
            self.cursor.execute(select_sql, (vector_id,))
            result = self.cursor.fetchone()
            
            if result:
                return np.array(result[self.vector_column], dtype=np.float32)
            return None
        except Exception as e:
            logger.error(f"获取向量失败: {e}")
            return None
    
    def get_all_vectors(self) -> Tuple[List[int], np.ndarray]:
        """
        获取所有向量
        
        Returns:
            Tuple[List[int], np.ndarray]: (ID列表, 向量数组)
        """
        try:
            select_sql = f"SELECT id, {self.vector_column} FROM {self.table_name} ORDER BY id"
            self.cursor.execute(select_sql)
            rows = self.cursor.fetchall()
            
            if not rows:
                return [], np.array([])
            
            ids = [row['id'] for row in rows]
            vectors = np.vstack([np.array(row[self.vector_column], dtype=np.float32) for row in rows])
            
            logger.info(f"获取到 {len(ids)} 条向量记录")
            return ids, vectors
        except Exception as e:
            logger.error(f"获取所有向量失败: {e}")
            return [], np.array([])
    
    def delete_vector(self, vector_id: int, commit: bool = True) -> bool:
        """
        删除指定ID的向量
        
        Args:
            vector_id: 向量ID
            commit: 是否立即提交
            
        Returns:
            bool: 删除是否成功
        """
        try:
            delete_sql = f"DELETE FROM {self.table_name} WHERE id = %s"
            self.cursor.execute(delete_sql, (vector_id,))
            
            if commit:
                self.conn.commit()
            
            if self.cursor.rowcount > 0:
                logger.info(f"向量ID {vector_id} 删除成功")
                return True
            else:
                logger.warning(f"向量ID {vector_id} 不存在")
                return False
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            self.conn.rollback()
            return False
    
    def delete_vectors_batch(self, vector_ids: List[int], commit: bool = True) -> int:
        """
        批量删除向量
        
        Args:
            vector_ids: 向量ID列表
            commit: 是否立即提交
            
        Returns:
            int: 成功删除的数量
        """
        try:
            if not vector_ids:
                return 0
            
            delete_sql = f"DELETE FROM {self.table_name} WHERE id = ANY(%s)"
            self.cursor.execute(delete_sql, (vector_ids,))
            deleted_count = self.cursor.rowcount
            
            if commit:
                self.conn.commit()
            
            logger.info(f"批量删除完成，删除了 {deleted_count} 条记录")
            return deleted_count
        except Exception as e:
            logger.error(f"批量删除向量失败: {e}")
            self.conn.rollback()
            return 0
    
    def clear_table(self, confirm: bool = False) -> bool:
        """
        清空表中所有数据
        
        Args:
            confirm: 确认删除标志
            
        Returns:
            bool: 清空是否成功
        """
        if not confirm:
            logger.warning("清空表需要确认，请设置 confirm=True")
            return False
        
        try:
            self.cursor.execute(f"TRUNCATE TABLE {self.table_name}")
            self.conn.commit()
            logger.info(f"表 {self.table_name} 已清空")
            return True
        except Exception as e:
            logger.error(f"清空表失败: {e}")
            self.conn.rollback()
            return False
    
    def get_vector_count(self) -> int:
        """
        获取向量总数
        
        Returns:
            int: 向量总数
        """
        try:
            count_sql = f"SELECT COUNT(*) as count FROM {self.table_name}"
            self.cursor.execute(count_sql)
            result = self.cursor.fetchone()
            return result['count']
        except Exception as e:
            logger.error(f"获取向量总数失败: {e}")
            return 0
    
    def create_ivfflat_index(self, lists: int = 100, 
                           ops: str = 'vector_cosine_ops') -> bool:
        """
        创建IVFFlat索引
        
        Args:
            lists: 聚类中心数量
            ops: 操作符类型
            
        Returns:
            bool: 创建是否成功
        """
        try:
            index_name = f"{self.table_name}_{self.vector_column}_ivfflat_idx"
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name} 
            ON {self.table_name} 
            USING ivfflat ({self.vector_column} {ops}) 
            WITH (lists = {lists})
            """
            self.cursor.execute(create_index_sql)
            self.conn.commit()
            logger.info(f"IVFFlat索引 {index_name} 创建成功")
            return True
        except Exception as e:
            logger.error(f"创建IVFFlat索引失败: {e}")
            self.conn.rollback()
            return False
    
    def create_hnsw_index(self, m: int = 16, ef_construction: int = 64,
                         ops: str = 'vector_cosine_ops') -> bool:
        """
        创建HNSW索引
        
        Args:
            m: 每个节点的最大连接数
            ef_construction: 构建时的搜索候选数
            ops: 操作符类型
            
        Returns:
            bool: 创建是否成功
        """
        try:
            index_name = f"{self.table_name}_{self.vector_column}_hnsw_idx"
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name} 
            ON {self.table_name} 
            USING hnsw ({self.vector_column} {ops}) 
            WITH (m = {m}, ef_construction = {ef_construction})
            """
            self.cursor.execute(create_index_sql)
            self.conn.commit()
            logger.info(f"HNSW索引 {index_name} 创建成功")
            return True
        except Exception as e:
            logger.error(f"创建HNSW索引失败: {e}")
            self.conn.rollback()
            return False
    
    def build_faiss_index(self, index_type: str = 'hnsw', 
                         m: int = 16, ef_construction: int = 200) -> bool:
        """
        构建FAISS索引（内存中）
        
        Args:
            index_type: 索引类型 ('hnsw', 'flat', 'ivf')
            m: HNSW参数m
            ef_construction: HNSW构建参数
            
        Returns:
            bool: 构建是否成功
        """
        try:
            ids, vectors = self.get_all_vectors()
            if len(ids) == 0:
                logger.warning("没有向量数据，无法构建FAISS索引")
                return False
            
            if index_type.lower() == 'hnsw':
                self.faiss_index = faiss.IndexHNSWFlat(self.vector_dim, m)
                self.faiss_index.hnsw.efConstruction = ef_construction
            elif index_type.lower() == 'flat':
                self.faiss_index = faiss.IndexFlatL2(self.vector_dim)
            elif index_type.lower() == 'ivf':
                quantizer = faiss.IndexFlatL2(self.vector_dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.vector_dim, min(100, len(ids)//10))
                self.faiss_index.train(vectors)
            else:
                raise ValueError(f"不支持的索引类型: {index_type}")
            
            self.faiss_index.add(vectors)
            self.id_mapping = ids
            
            logger.info(f"FAISS {index_type.upper()} 索引构建成功，包含 {len(ids)} 个向量")
            return True
        except Exception as e:
            logger.error(f"构建FAISS索引失败: {e}")
            return False
    
    def search_similar_vectors(self, query_vector: Union[List, np.ndarray], 
                             k: int = 10, use_faiss: bool = False) -> List[Tuple[int, float]]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            k: 返回top-k结果
            use_faiss: 是否使用FAISS索引
            
        Returns:
            List[Tuple[int, float]]: (向量ID, 距离)的列表
        """
        try:
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector, dtype=np.float32)
            
            if use_faiss and self.faiss_index is not None:
                # 使用FAISS搜索
                distances, indices = self.faiss_index.search(query_vector.reshape(1, -1), k)
                results = [(self.id_mapping[idx], float(dist)) 
                          for idx, dist in zip(indices[0], distances[0]) if idx != -1]
            else:
                # 使用PostgreSQL向量搜索
                search_sql = f"""
                SELECT id, {self.vector_column} <-> %s as distance 
                FROM {self.table_name} 
                ORDER BY distance 
                LIMIT %s
                """
                self.cursor.execute(search_sql, (query_vector.tolist(), k))
                rows = self.cursor.fetchall()
                results = [(row['id'], float(row['distance'])) for row in rows]
            
            logger.info(f"搜索到 {len(results)} 个相似向量")
            return results
        except Exception as e:
            logger.error(f"搜索相似向量失败: {e}")
            return []
    
    def compute_in_degrees(self) -> Dict[int, int]:
        """
        计算FAISS图中每个节点的入度
        
        Returns:
            Dict[int, int]: {向量ID: 入度}
        """
        try:
            if self.faiss_index is None or not hasattr(self.faiss_index, 'hnsw'):
                logger.error("需要先构建HNSW索引")
                return {}
            
            hnsw = self.faiss_index.hnsw
            neighbors = faiss.vector_to_array(hnsw.neighbors).astype(np.int32)
            offsets = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
            
            N = len(self.id_mapping)
            end = int(offsets[N])
            valid_neighbors = neighbors[:end]
            valid_neighbors = valid_neighbors[valid_neighbors >= 0]
            
            counts = np.bincount(valid_neighbors, minlength=N)
            in_degrees = {self.id_mapping[i]: int(counts[i]) for i in range(N)}
            
            logger.info(f"计算了 {len(in_degrees)} 个向量的入度")
            return in_degrees
        except Exception as e:
            logger.error(f"计算入度失败: {e}")
            return {}
    
    def get_low_in_degree_vectors(self, top_k: int = 50) -> List[int]:
        """
        获取低入度向量ID列表
        
        Args:
            top_k: 返回前k个低入度向量
            
        Returns:
            List[int]: 低入度向量ID列表
        """
        in_degrees = self.compute_in_degrees()
        if not in_degrees:
            return []
        
        # 按入度排序，返回最低的top_k个
        sorted_ids = sorted(in_degrees.items(), key=lambda x: x[1])[:top_k]
        result = [vid for vid, _ in sorted_ids]
        
        logger.info(f"获取到 {len(result)} 个低入度向量")
        return result
    
    def get_statistics(self) -> Dict:
        """
        获取数据库统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            stats = {
                'total_vectors': self.get_vector_count(),
                'vector_dimension': self.vector_dim,
                'table_name': self.table_name,
                'vector_column': self.vector_column,
                'has_faiss_index': self.faiss_index is not None,
                'faiss_index_type': type(self.faiss_index).__name__ if self.faiss_index else None
            }
            
            # 获取表大小
            try:
                size_sql = f"""
                SELECT pg_size_pretty(pg_total_relation_size('{self.table_name}')) as table_size
                """
                self.cursor.execute(size_sql)
                result = self.cursor.fetchone()
                stats['table_size'] = result['table_size']
            except:
                stats['table_size'] = 'Unknown'
            
            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def __repr__(self):
        return f"PgVectorClient(table={self.table_name}, dim={self.vector_dim})"