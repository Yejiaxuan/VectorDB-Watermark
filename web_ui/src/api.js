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
 * 在指定表的向量列中嵌入水印并自动下载ID文件
 * @param {{host,port,dbname,user,password}} dbParams 数据库连接参数
 * @param {string} table 表名
 * @param {string} idColumn 主键列名
 * @param {string} vectorColumn 向量列名
 * @param {string} message 水印消息
 * @returns {Promise<{success: boolean, message: string, updated: number, file_id: string}>} 结果
 */
export async function embedWatermark(dbParams, table, idColumn, vectorColumn, message) {
  const res = await fetch('/api/embed_watermark', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      db_params: dbParams,
      table,
      id_column: idColumn,
      vector_column: vectorColumn,
      message,
      total_vecs: 1600  // 默认使用的向量数量
    }),
  });
  
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || '水印嵌入失败');
  }
  
  const result = await res.json();
  
  // 如果嵌入成功且有文件ID，自动触发下载
  if (result.success && result.file_id) {
    try {
      await downloadIdsFileById(result.file_id, table, vectorColumn);
    } catch (error) {
      console.error('自动下载ID文件失败，请稍后手动下载', error);
      // 继续返回原始结果，但添加警告
      result.downloadWarning = '自动下载ID文件失败，请稍后手动下载';
    }
  }
  
  return result;
}

/**
 * 通过文件ID下载水印ID文件
 * @param {string} fileId 文件ID
 * @param {string} table 表名 (用于生成下载文件名)
 * @param {string} vectorColumn 向量列名 (用于生成下载文件名)
 * @returns {Promise<boolean>}
 */
export async function downloadIdsFileById(fileId, table, vectorColumn) {
  try {
    const response = await fetch(`/api/download_ids_file/${fileId}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('找不到ID文件，可能已过期或被删除');
      }
      const error = await response.json();
      throw new Error(error.detail || '下载ID文件失败');
    }
    
    // 获取blob数据
    const blob = await response.blob();
    
    // 创建下载链接
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    // 使用更友好的文件名
    a.download = `向量水印_${table}_${vectorColumn}_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(a);
    a.click();
    
    // 清理
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    return true;
  } catch (error) {
    console.error('下载ID文件错误:', error);
    throw error;
  }
}

/**
 * 使用上传的ID文件提取水印
 * @param {{host,port,dbname,user,password}} dbParams 数据库连接参数
 * @param {string} table 表名
 * @param {string} idColumn 主键列名
 * @param {string} vectorColumn 向量列名
 * @param {File} idsFile 上传的ID文件
 * @returns {Promise<{success: boolean, message: string, blocks: number, recovered: number}>} 结果
 */
export async function extractWatermarkWithFile(dbParams, table, idColumn, vectorColumn, idsFile) {
  // 创建FormData对象
  const formData = new FormData();
  formData.append('file', idsFile);
  formData.append('db_json', JSON.stringify(dbParams));
  formData.append('table', table);
  formData.append('id_column', idColumn);
  formData.append('vector_column', vectorColumn);
  
  const res = await fetch('/api/extract_watermark_with_file', {
    method: 'POST',
    body: formData
  });
  
  if (!res.ok) {
    let errorMessage = '水印提取失败';
    try {
      const err = await res.json();
      errorMessage = err.detail || errorMessage;
    } catch (e) {
      // 如果解析JSON失败，使用默认错误消息
    }
    throw new Error(errorMessage);
  }
  
  return res.json();
}