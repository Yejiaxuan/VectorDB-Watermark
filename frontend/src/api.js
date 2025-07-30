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
 * 获取向量维度
 * @param {{host,port,dbname,user,password}} dbParams
 * @param {string} table
 * @param {string} vectorColumn
 * @returns {Promise<{dimension: number}>}
 */
export async function getVectorDimension(dbParams, table, vectorColumn) {
  const res = await fetch(`/api/get_vector_dimension?table=${encodeURIComponent(table)}&vector_column=${encodeURIComponent(vectorColumn)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '获取向量维度失败');
  }
  return res.json();
}

/**
 * 检查模型是否存在
 * @param {number} dimension
 * @returns {Promise<{exists: boolean, model_path: string, dimension: number}>}
 */
export async function checkModel(dimension) {
  const res = await fetch(`/api/check_model?dimension=${dimension}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '检查模型失败');
  }
  return res.json();
}

/**
 * 训练模型
 * @param {{host,port,dbname,user,password}} dbParams
 * @param {string} table
 * @param {string} vectorColumn
 * @param {number} dimension
 * @param {Object} trainParams - 训练参数
 * @param {number} trainParams.epochs - 训练轮数
 * @param {number} trainParams.learning_rate - 学习率
 * @param {number} trainParams.batch_size - 批处理大小
 * @param {number} trainParams.val_ratio - 验证集比例
 * @returns {Promise<{task_id: string, message: string, dimension: number, train_params: Object}>}
 */
export async function trainModel(dbParams, table, vectorColumn, dimension, trainParams = {}) {
  const {
    epochs = 100,
    learning_rate = 0.0003,
    batch_size = 8192,
    val_ratio = 0.15
  } = trainParams;

  const res = await fetch(`/api/train_model?table=${encodeURIComponent(table)}&vector_column=${encodeURIComponent(vectorColumn)}&dimension=${dimension}&epochs=${epochs}&learning_rate=${learning_rate}&batch_size=${batch_size}&val_ratio=${val_ratio}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '启动训练失败');
  }
  return res.json();
}

/**
 * 获取训练状态
 * @param {string} taskId
 * @returns {Promise<{status: string, message: string, progress: number}>}
 */
export async function getTrainingStatus(taskId) {
  const res = await fetch(`/api/training_status/${taskId}`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '获取训练状态失败');
  }
  return res.json();
}

/**
 * 在指定表的向量列中嵌入水印
 * @param {{host,port,dbname,user,password}} dbParams 数据库连接参数
 * @param {string} table 表名
 * @param {string} idColumn 主键列名
 * @param {string} vectorColumn 向量列名
 * @param {string} message 明文消息（16字节）
 * @param {number} embedRate 水印嵌入率（0-1之间的浮点数），默认0.1（10%）
 * @param {string} encryptionKey AES-GCM加密密钥
 * @returns {Promise<{success: boolean, message: string, updated: number}>} 结果
 */
export async function embedWatermark(dbParams, table, idColumn, vectorColumn, message, embedRate = 0.1, encryptionKey) {
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
      encryption_key: encryptionKey,
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
 * @param {string} encryptionKey AES-GCM解密密钥
 * @param {string} nonce nonce的十六进制表示，用于解密
 * @returns {Promise<{success: boolean, message: string, blocks: number, recovered: number}>} 结果
 */
export async function extractWatermark(dbParams, table, idColumn, vectorColumn, embedRate = 0.1, encryptionKey, nonce) {
  const res = await fetch('/api/extract-watermark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      db_params: dbParams,
      table,
      id_column: idColumn,
      vector_column: vectorColumn,
      embed_rate: embedRate,
      encryption_key: encryptionKey,
      nonce
    })
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '水印提取失败');
  }
  
  return res.json();
}


/**
 * 获取向量降维可视化数据
 * @param {Array} originalVectors 原始向量数组
 * @param {Array} embeddedVectors 嵌入水印后的向量数组
 * @param {string} method 降维方法 (tsne 或 pca)
 * @returns {Promise<Object>} 降维结果
 */
export async function getVectorVisualization(originalVectors, embeddedVectors, method = 'tsne', useAllSamples = true) {
  const res = await fetch('/api/vector_visualization', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      original_vectors: originalVectors,
      embedded_vectors: embeddedVectors,
      method,
      use_all_samples: useAllSamples
    })
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '降维处理失败');
  }
  
  return res.json();
}


/**
 * 异步获取向量可视化数据
 * @param {Array} originalVectors 原始向量数组
 * @param {Array} embeddedVectors 嵌入水印后的向量数组
 * @param {string} method 降维方法 ('tsne' 或 'pca')
 * @param {boolean} useAllSamples 是否使用所有样本
 * @returns {Promise<Object>} 包含任务ID和预估时间的响应
 */
export async function getVectorVisualizationAsync(originalVectors, embeddedVectors, method = 'tsne', useAllSamples = true) {
  const res = await fetch('/api/vector_visualization_async', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      original_vectors: originalVectors,
      embedded_vectors: embeddedVectors,
      method,
      use_all_samples: useAllSamples
    })
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '启动降维处理失败');
  }
  
  return res.json();
}

/**
 * 获取可视化处理任务的状态
 * @param {string} taskId 任务ID
 * @returns {Promise<Object>} 任务状态信息
 */
export async function getVisualizationStatus(taskId) {
  const res = await fetch(`/api/visualization_status/${taskId}`);
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '获取可视化状态失败');
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
 * @param {string} message 明文消息（16字节）
 * @param {number} embedRate 水印嵌入率（0-1之间的浮点数），默认0.1（10%）
 * @param {string} encryptionKey AES-GCM加密密钥
 * @returns {Promise<{success: boolean, message: string, updated: number, nonce: string}>} 结果
 */
export async function embedMilvusWatermark(dbParams, collectionName, idField, vectorField, message, embedRate = 0.1, encryptionKey) {
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
      encryption_key: encryptionKey,
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
 * @param {string} encryptionKey AES-GCM解密密钥
 * @param {string} nonce nonce的十六进制表示，用于解密
 * @returns {Promise<{success: boolean, message: string, blocks: number, recovered: number}>} 结果
 */
export async function extractMilvusWatermark(dbParams, collectionName, idField, vectorField, embedRate = 0.1, encryptionKey, nonce) {
  const res = await fetch('/api/milvus/extract_watermark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      db_params: dbParams,
      collection_name: collectionName,
      id_field: idField,
      vector_field: vectorField,
      embed_rate: embedRate,
      encryption_key: encryptionKey,
      nonce
    })
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Milvus水印提取失败');
  }
  
  return res.json();
}

/**
 * 获取Milvus向量维度
 * @param {{host: string, port: number}} dbParams
 * @param {string} collectionName
 * @param {string} vectorField
 * @returns {Promise<{dimension: number}>}
 */
export async function getMilvusVectorDimension(dbParams, collectionName, vectorField) {
  const res = await fetch(`/api/milvus/get_vector_dimension?collection_name=${encodeURIComponent(collectionName)}&vector_field=${encodeURIComponent(vectorField)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '获取Milvus向量维度失败');
  }
  return res.json();
}

/**
 * 训练Milvus模型
 * @param {{host: string, port: number}} dbParams
 * @param {string} collectionName
 * @param {string} vectorField
 * @param {number} dimension
 * @param {Object} trainParams - 训练参数
 * @param {number} trainParams.epochs - 训练轮数
 * @param {number} trainParams.learning_rate - 学习率
 * @param {number} trainParams.batch_size - 批处理大小
 * @param {number} trainParams.val_ratio - 验证集比例
 * @returns {Promise<{task_id: string, message: string, dimension: number, train_params: Object}>}
 */
export async function trainMilvusModel(dbParams, collectionName, vectorField, dimension, trainParams = {}) {
  const {
    epochs = 100,
    learning_rate = 0.0003,
    batch_size = 8192,
    val_ratio = 0.15
  } = trainParams;

  const res = await fetch(`/api/milvus/train_model?collection_name=${encodeURIComponent(collectionName)}&vector_field=${encodeURIComponent(vectorField)}&dimension=${dimension}&epochs=${epochs}&learning_rate=${learning_rate}&batch_size=${batch_size}&val_ratio=${val_ratio}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '启动Milvus训练失败');
  }
  return res.json();
}


/**
 * 获取Milvus向量降维可视化数据
 * @param {Array} originalVectors 原始向量数组
 * @param {Array} embeddedVectors 嵌入水印后的向量数组
 * @param {string} method 降维方法 (tsne 或 pca)
 * @param {boolean} useAllSamples 是否使用所有样本
 * @returns {Promise<Object>} 降维结果
 */
export async function getMilvusVectorVisualization(originalVectors, embeddedVectors, method = 'tsne', useAllSamples = true) {
  const res = await fetch('/api/milvus/vector_visualization', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      original_vectors: originalVectors,
      embedded_vectors: embeddedVectors,
      method,
      use_all_samples: useAllSamples
    })
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Milvus降维处理失败');
  }
  
  return res.json();
}

/**
 * 异步获取Milvus向量可视化数据
 * @param {Array} originalVectors 原始向量数组
 * @param {Array} embeddedVectors 嵌入水印后的向量数组
 * @param {string} method 降维方法 ('tsne' 或 'pca')
 * @param {boolean} useAllSamples 是否使用所有样本
 * @returns {Promise<Object>} 包含任务ID和预估时间的响应
 */
export async function getMilvusVectorVisualizationAsync(originalVectors, embeddedVectors, method = 'tsne', useAllSamples = true) {
  const res = await fetch('/api/milvus/vector_visualization_async', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      original_vectors: originalVectors,
      embedded_vectors: embeddedVectors,
      method,
      use_all_samples: useAllSamples
    })
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '启动Milvus降维处理失败');
  }
  
  return res.json();
}

/**
 * 获取Milvus可视化处理任务的状态
 * @param {string} taskId 任务ID
 * @returns {Promise<Object>} 任务状态信息
 */
export async function getMilvusVisualizationStatus(taskId) {
  const res = await fetch(`/api/milvus/visualization_status/${taskId}`);
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '获取Milvus可视化状态失败');
  }
  
  return res.json();
}