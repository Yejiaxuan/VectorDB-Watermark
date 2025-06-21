import React, { useState, useEffect, useRef } from 'react';
import {
  connectDB,
  fetchTables,
  fetchColumns,
  fetchPrimaryKeys,
  embedWatermark,
  extractWatermarkWithFile
} from '../api';

export default function PgvectorPage() {
  // —— 数据库连接状态 ——  
  const [ip, setIp]           = useState('localhost');
  const [port, setPort]       = useState(5432);
  const [dbname, setDbname]   = useState('test');
  const [user, setUser]       = useState('postgres');
  const [password, setPassword] = useState('');
  const [connected, setConnected] = useState(false);
  const [statusMsg, setStatusMsg] = useState('未连接');
  const [loadingConn, setLoadingConn] = useState(false);

  // —— 表/列 列表 ——  
  const [tables, setTables]   = useState([]);
  const [table, setTable]     = useState('');
  const [primaryKeys, setPrimaryKeys] = useState([]);
  const [primaryKey, setPrimaryKey] = useState('');
  const [columns, setColumns] = useState([]);
  const [column, setColumn]   = useState('');

  // —— 水印操作状态 ——  
  const [message, setMessage] = useState('ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEF');
  const [embedResult, setEmbedResult] = useState('');
  const [extractResult, setExtractResult] = useState('');
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [fileId, setFileId] = useState(''); // 保存最后嵌入的文件ID
  
  // —— 文件上传相关 ——
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const fileInputRef = useRef(null);
  

  // 连接数据库
  const handleConnect = async () => {
    setLoadingConn(true);
    setStatusMsg('连接中…');
    try {
      const { success, message: msg } = await connectDB({ host: ip, port, dbname, user, password });
      setConnected(success);
      setStatusMsg(success ? msg : '连接失败');
    } catch (err) {
      setConnected(false);
      setStatusMsg(`错误：${err.message}`);
    } finally {
      setLoadingConn(false);
    }
  };

  // 拉表列表
  useEffect(() => {
    if (connected) {
      fetchTables({ host: ip, port, dbname, user, password })
        .then(ts => {
          setTables(ts);
          if (ts.length) setTable(ts[0]);
        })
        .catch(err => console.error('拉表失败', err));
    } else {
      setTables([]); setTable(''); setPrimaryKeys([]); setPrimaryKey(''); setColumns([]); setColumn('');
    }
  }, [connected]);

  // 当表格或列变更时，重置水印状态
  useEffect(() => {
    setFileId(''); // 重置文件ID
    setEmbedResult('');
    setExtractResult('');
  }, [table, column]);

  // 当选择表后获取主键列
  useEffect(() => {
    if (connected && table) {
      fetchPrimaryKeys({ host: ip, port, dbname, user, password }, table)
        .then(keys => {
          setPrimaryKeys(keys);
          if (keys.length) setPrimaryKey(keys[0]);
          else setPrimaryKey('');
        })
        .catch(err => console.error('获取主键失败', err));
    } else {
      setPrimaryKeys([]); 
      setPrimaryKey('');
    }
  }, [connected, table]);

  // 拉列列表
  useEffect(() => {
    if (connected && table) {
      fetchColumns({ host: ip, port, dbname, user, password }, table)
        .then(cs => {
          setColumns(cs);
          if (cs.length) setColumn(cs[0]);
        })
        .catch(err => console.error('拉列失败', err));
    } else {
      setColumns([]); setColumn('');
    }
  }, [connected, table]);

  // 嵌入水印
  const handleEmbed = async () => {
    // 验证必要条件
    if (!connected || !message || !table || !column || !primaryKey || message.length !== 32) return;
    
    // 设置加载状态和清除之前的结果
    setIsEmbedding(true);
    setEmbedResult('嵌入水印中...');
    setExtractResult('');
    setFileId(''); // 重置文件ID
    
    try {
      // 调用API函数
      const dbParams = { host: ip, port, dbname, user, password };
      const result = await embedWatermark(dbParams, table, primaryKey, column, message);
      
      // 更新UI
      setEmbedResult(`成功: ${result.message}。ID文件已自动下载，请妥善保存用于提取水印。`);
      
      // 保存文件ID供参考
      if (result.file_id) {
        setFileId(result.file_id);
      }
      
      // 检查是否有下载警告
      if (result.downloadWarning) {
        setEmbedResult(prev => `${prev}\n警告: ${result.downloadWarning}`);
      }
    } catch (error) {
      setEmbedResult(`错误: ${error.message}`);
      console.error('水印嵌入错误', error);
    } finally {
      setIsEmbedding(false);
    }
  };

  // 提取水印
  const handleExtract = async () => {
    if (!connected || !table || !column || !primaryKey || !selectedFile) return;
    
    setIsExtracting(true);
    setExtractResult('提取水印中...');
    
    try {
      const dbParams = { host: ip, port, dbname, user, password };
      
      // 使用上传的ID文件提取
      const result = await extractWatermarkWithFile(dbParams, table, primaryKey, column, selectedFile);
      
      // 处理结果
      if (result.success) {
        setExtractResult(`提取成功：${result.message} (恢复 ${result.recovered}/${result.blocks} 个区块)`);
      } else {
        setExtractResult(`提取失败：${result.error}`);
      }
    } catch (error) {
      setExtractResult(`错误: ${error.message}`);
      console.error('水印提取错误', error);
    } finally {
      setIsExtracting(false);
    }
  };
  
  // 处理文件选择
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setFileName(file.name);
    }
  };
  
  // 清除选择的文件
  const clearFileSelection = () => {
    setSelectedFile(null);
    setFileName('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="p-8 grid grid-cols-2 gap-8">
      {/* 左侧：连接配置 */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-2xl font-semibold mb-4">PGVector 连接配置</h2>
        <div className="space-y-3">
          {/** Host/IP **/}
          <div>
            <label className="block font-medium">Host / IP</label>
            <input
              type="text"
              value={ip}
              onChange={e => setIp(e.target.value)}
              className="w-full border rounded px-3 py-2"
              disabled={loadingConn}
            />
          </div>
          {/** Port **/}
          <div>
            <label className="block font-medium">Port</label>
            <input
              type="number"
              value={port}
              onChange={e => setPort(parseInt(e.target.value, 10) || 0)}
              className="w-full border rounded px-3 py-2"
              disabled={loadingConn}
            />
          </div>
          {/** Database **/}
          <div>
            <label className="block font-medium">Database</label>
            <input
              type="text"
              value={dbname}
              onChange={e => setDbname(e.target.value)}
              className="w-full border rounded px-3 py-2"
              disabled={loadingConn}
            />
          </div>
          {/** User **/}
          <div>
            <label className="block font-medium">User</label>
            <input
              type="text"
              value={user}
              onChange={e => setUser(e.target.value)}
              className="w-full border rounded px-3 py-2"
              disabled={loadingConn}
            />
          </div>
          {/** Password **/}
          <div>
            <label className="block font-medium">Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full border rounded px-3 py-2"
              disabled={loadingConn}
            />
          </div>
          {/** Connect 按钮 **/}
          <button
            onClick={handleConnect}
            disabled={loadingConn}
            className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded disabled:opacity-50"
          >
            {loadingConn
              ? '连接中…'
              : connected
              ? '已连接'
              : '连接数据库'}
          </button>
        </div>
        <p className="mt-4 text-sm text-gray-600">状态：{statusMsg}</p>
      </div>

      {/* 右侧：水印＆表列选择 */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-2xl font-semibold mb-4">水印嵌入与提取</h2>

        {/* 先选表 & 列 */}
        {connected && (
          <div className="space-y-4 mb-6">
            <div>
              <label className="block font-medium">选择表</label>
              <select
                value={table}
                onChange={e => setTable(e.target.value)}
                className="w-full border rounded px-3 py-2"
              >
                {tables.map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            
            {/* 添加主键选择下拉框 */}
            <div>
              <label className="block font-medium">选择主键列</label>
              <select
                value={primaryKey}
                onChange={e => setPrimaryKey(e.target.value)}
                className="w-full border rounded px-3 py-2"
              >
                {primaryKeys.length === 0 ? (
                  <option value="">该表无主键</option>
                ) : (
                  primaryKeys.map(pk => (
                    <option key={pk} value={pk}>{pk}</option>
                  ))
                )}
              </select>
            </div>
            
            <div>
              <label className="block font-medium">选择向量列</label>
              <select
                value={column}
                onChange={e => setColumn(e.target.value)}
                className="w-full border rounded px-3 py-2"
              >
                {columns.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
          </div>
        )}

        {/* 嵌入水印区域 */}
        <div className="space-y-3 pb-4 border-b">
          <h3 className="font-semibold mb-2">嵌入水印</h3>
          <div>
            <label className="block font-medium">消息内容 (32字符)</label>
            <textarea
              rows={2}
              value={message}
              onChange={e => setMessage(e.target.value)}
              className="w-full border rounded px-3 py-2"
              disabled={!connected || !table || !column}
              maxLength={32}
            />
            <p className="text-xs text-gray-500 mt-1">
              {message.length}/32 字符 {message.length !== 32 && message.length > 0 && "(需要恰好32个字符)"}
            </p>
          </div>

          {/* 嵌入按钮 */}
          <button
            onClick={handleEmbed}
            disabled={!connected || !table || !primaryKey || !column || !message || message.length !== 32 || isEmbedding}
            className="w-full bg-green-500 hover:bg-green-600 text-white py-2 rounded disabled:opacity-50"
          >
            {isEmbedding ? '嵌入中...' : '嵌入水印并下载ID文件'}
          </button>
          
          {/* 嵌入结果显示 */}
          {embedResult && (
            <div className="mt-2 p-3 bg-gray-100 rounded">
              <h3 className="font-semibold mb-1">嵌入结果:</h3>
              <p className="whitespace-pre-line">{embedResult}</p>
            </div>
          )}
        </div>
          
        {/* 水印提取区域 */}
        <div className="mt-4 pt-2">
          <h3 className="font-semibold mb-3">水印提取</h3>
          
          {/* 文件上传区域 */}
          <div className="mb-3 border rounded-lg p-3 bg-gray-50">
            <div className="flex items-center justify-between">
              <label className="block font-medium mb-2">上传ID文件:</label>
              {selectedFile && (
                <button
                  onClick={clearFileSelection}
                  className="text-red-600 hover:text-red-800 text-sm"
                >
                  清除
                </button>
              )}
            </div>
            
            <div className="flex items-center space-x-2">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".json"
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4
                  file:rounded file:border-0
                  file:text-sm file:font-semibold
                  file:bg-purple-50 file:text-purple-700
                  hover:file:bg-purple-100"
              />
            </div>
            
            {fileName && (
              <p className="mt-1 text-sm text-gray-600">
                已选择: {fileName}
              </p>
            )}

            <div className="mt-3 p-2 bg-blue-50 border border-blue-200 rounded text-sm text-blue-800">
              提示：请上传之前嵌入水印时下载的ID文件（JSON格式）
            </div>
          </div>
          
          {/* 提取按钮 */}
          <button
            onClick={handleExtract}
            disabled={!connected || !table || !primaryKey || !column || isExtracting || !selectedFile}
            className="w-full bg-purple-500 hover:bg-purple-600 text-white py-2 rounded disabled:opacity-50"
          >
            {isExtracting ? '提取中...' : '提取水印'}
          </button>

          {/* 提取结果显示 */}
          {extractResult && (
            <div className="mt-3 p-3 bg-gray-100 rounded">
              <h3 className="font-semibold mb-1">提取结果:</h3>
              <p>{extractResult}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}