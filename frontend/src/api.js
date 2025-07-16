// web_ui/src/api.js
export async function connectDB(dbParams) {
  const res = await fetch("/api/connect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "连接失败");
  }
  return res.json();  // { success: true, message: "连接成功" }
}

/**
 * 获取所有 public schema 下的表名
 * @param {{host,port,dbname,user,password}} dbParams
 * @returns {Promise<string[]>}
 */
export async function fetchTables(dbParams) {
  const res = await fetch('/api/tables', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '拉表失败');
  }
  const { tables } = await res.json();
  return tables;
}

/**
 * 获取指定表的所有 vector 类型列
 * @param {{host,port,dbname,user,password}} dbParams
 * @param {string} table
 * @returns {Promise<string[]>}
 */
export async function fetchColumns(dbParams, table) {
  const res = await fetch(`/api/columns?table=${encodeURIComponent(table)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '拉列失败');
  }
  const { columns } = await res.json();
  return columns;
}

/**
 * 获取指定表的主键列
 * @param {{host,port,dbname,user,password}} dbParams
 * @param {string} table
 * @returns {Promise<string[]>}
 */
export async function fetchPrimaryKeys(dbParams, table) {
  const res = await fetch(`/api/primarykeys?table=${encodeURIComponent(table)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '获取主键失败');
  }
  const { keys } = await res.json();
  return keys;
}

/**
 * 在指定表的向量列中嵌入水印
 * @param {{host,port,dbname,user,password}} dbParams 数据库连接参数
 * @param {string} table 表名
 * @param {string} idColumn 主键列名
 * @param {string} vectorColumn 向量列名
 * @param {string} message 水印消息
 * @param {number} embedRate 水印嵌入率（0-1之间的浮点数），默认0.1（10%）
 * @returns {Promise<{success: boolean, message: string, updated: number}>} 结果
 */
export async function embedWatermark(dbParams, table, idColumn, vectorColumn, message, embedRate = 0.1) {
  const res = await fetch('/api/embed_watermark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      db_params: dbParams,
      table,
      id_column: idColumn,
      vector_column: vectorColumn,
      message,
      embed_rate: embedRate,
      total_vecs: 1600  // 保留兼容性，但现在主要使用embed_rate
    }),
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '水印嵌入失败');
  }
  
  return res.json();
}



/**
 * 提取水印，重新计算低入度节点
 * @param {{host,port,dbname,user,password}} dbParams 数据库连接参数
 * @param {string} table 表名
 * @param {string} idColumn 主键列名
 * @param {string} vectorColumn 向量列名
 * @param {number} embedRate 水印嵌入率（0-1之间的浮点数），默认0.1（10%）
 * @returns {Promise<{success: boolean, message: string, blocks: number, recovered: number}>} 结果
 */
export async function extractWatermark(dbParams, table, idColumn, vectorColumn, embedRate = 0.1) {
  const res = await fetch('/api/extract-watermark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      db_params: dbParams,
      table,
      id_column: idColumn,
      vector_column: vectorColumn,
      embed_rate: embedRate
    })
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '水印提取失败');
  }
  
  return res.json();
}



// ===== Milvus API 函数 =====

/**
 * 连接Milvus数据库
 * @param {{host: string, port: number}} dbParams
 * @returns {Promise<{success: boolean, message: string}>}
 */
export async function connectMilvusDB(dbParams) {
  const res = await fetch("/api/milvus/connect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Milvus连接失败");
  }
  return res.json();
}

/**
 * 获取所有Milvus集合
 * @param {{host: string, port: number}} dbParams
 * @returns {Promise<string[]>}
 */
export async function fetchMilvusCollections(dbParams) {
  const res = await fetch('/api/milvus/collections', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '获取集合失败');
  }
  const { collections } = await res.json();
  return collections;
}

/**
 * 获取指定集合的所有向量字段
 * @param {{host: string, port: number}} dbParams
 * @param {string} collectionName
 * @returns {Promise<string[]>}
 */
export async function fetchMilvusVectorFields(dbParams, collectionName) {
  const res = await fetch(`/api/milvus/vector_fields?collection_name=${encodeURIComponent(collectionName)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '获取向量字段失败');
  }
  const { fields } = await res.json();
  return fields;
}

/**
 * 获取指定集合的主键字段
 * @param {{host: string, port: number}} dbParams
 * @param {string} collectionName
 * @returns {Promise<string[]>}
 */
export async function fetchMilvusPrimaryKeys(dbParams, collectionName) {
  const res = await fetch(`/api/milvus/primary_keys?collection_name=${encodeURIComponent(collectionName)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '获取主键字段失败');
  }
  const { keys } = await res.json();
  return keys;
}

/**
 * 在指定Milvus集合的向量字段中嵌入水印
 * @param {{host: string, port: number}} dbParams Milvus连接参数
 * @param {string} collectionName 集合名
 * @param {string} idField 主键字段名
 * @param {string} vectorField 向量字段名
 * @param {string} message 水印消息
 * @param {number} embedRate 水印嵌入率（0-1之间的浮点数），默认0.1（10%）
 * @returns {Promise<{success: boolean, message: string, updated: number}>} 结果
 */
export async function embedMilvusWatermark(dbParams, collectionName, idField, vectorField, message, embedRate = 0.1) {
  const res = await fetch('/api/milvus/embed_watermark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      db_params: dbParams,
      collection_name: collectionName,
      id_field: idField,
      vector_field: vectorField,
      message,
      embed_rate: embedRate,
      total_vecs: 1600  // 保留兼容性
    }),
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Milvus水印嵌入失败');
  }
  
  return res.json();
}





/**
 * 从Milvus提取水印，重新计算低入度节点
 * @param {{host: string, port: number}} dbParams Milvus连接参数
 * @param {string} collectionName 集合名
 * @param {string} idField 主键字段名
 * @param {string} vectorField 向量字段名
 * @param {number} embedRate 水印嵌入率（0-1之间的浮点数），默认0.1（10%）
 * @returns {Promise<{success: boolean, message: string, blocks: number, recovered: number}>} 结果
 */
export async function extractMilvusWatermark(dbParams, collectionName, idField, vectorField, embedRate = 0.1) {
  const res = await fetch('/api/milvus/extract_watermark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      db_params: dbParams,
      collection_name: collectionName,
      id_field: idField,
      vector_field: vectorField,
      embed_rate: embedRate
    })
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Milvus水印提取失败');
  }
  
  return res.json();
}

