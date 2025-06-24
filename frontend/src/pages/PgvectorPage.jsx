import React, { useState, useEffect, useRef } from 'react';
import Toast from '../components/Toast';
import Combobox from '../components/Combobox';
import {
  connectDB,
  fetchTables,
  fetchColumns,
  fetchPrimaryKeys,
  embedWatermark,
  extractWatermarkWithFile
} from '../api';

export default function PgvectorPage() {
  // —— 步骤控制 ——
  const [currentStep, setCurrentStep] = useState(1);
  const [stepCompleted, setStepCompleted] = useState(false);
  
  // —— 数据库连接状态 ——  
  const [ip, setIp] = useState('localhost');
  const [port, setPort] = useState(5432);
  const [dbname, setDbname] = useState('test');
  const [user, setUser] = useState('postgres');
  const [password, setPassword] = useState('');
  const [connected, setConnected] = useState(false);
  const [statusMsg, setStatusMsg] = useState('未连接');
  const [loadingConn, setLoadingConn] = useState(false);

  // —— 表单验证 ——
  const [formErrors, setFormErrors] = useState({});

  // —— 表/列 列表 ——  
  const [tables, setTables] = useState([]);
  const [table, setTable] = useState('');
  const [primaryKeys, setPrimaryKeys] = useState([]);
  const [primaryKey, setPrimaryKey] = useState('');
  const [columns, setColumns] = useState([]);
  const [column, setColumn] = useState('');

  // —— Tab 控制 ——
  const [activeTab, setActiveTab] = useState('embed');

  // —— 水印操作状态 ——  
  const [message, setMessage] = useState('ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEF');
  const [embedResult, setEmbedResult] = useState('');
  const [extractResult, setExtractResult] = useState('');
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [fileId, setFileId] = useState('');
  
  // —— 文件上传相关 ——
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const fileInputRef = useRef(null);

  // —— Toast 相关 ——
  const [toasts, setToasts] = useState([]);

  // 显示Toast提示
  const showToast = (message, type = 'success') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type, isVisible: true }]);
  };

  // 移除Toast
  const removeToast = (id) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  // 表单验证
  const validateForm = () => {
    const errors = {};
    if (!ip.trim()) errors.ip = '请输入主机地址';
    if (!port || port < 1 || port > 65535) errors.port = '请输入有效端口号';
    if (!dbname.trim()) errors.dbname = '请输入数据库名称';
    if (!user.trim()) errors.user = '请输入用户名';
    
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  // 连接数据库
  const handleConnect = async () => {
    if (!validateForm()) return;
    
    setLoadingConn(true);
    setStatusMsg('连接中…');
    try {
      const { success, message: msg } = await connectDB({ host: ip, port, dbname, user, password });
      setConnected(success);
      setStatusMsg(success ? msg : '连接失败');
      if (success) {
        setStepCompleted(true);
        showToast('数据库连接成功！', 'success');
        // 连接成功后自动进入下一步
        setTimeout(() => setCurrentStep(2), 800);
      }
    } catch (err) {
      setConnected(false);
      setStatusMsg(`错误：${err.message}`);
      showToast(`连接失败：${err.message}`, 'error');
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
    setFileId('');
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
    if (!connected || !message || !table || !column || !primaryKey || message.length !== 32) return;
    
    setIsEmbedding(true);
    setEmbedResult('');
    setExtractResult('');
    setFileId('');
    
    try {
      const dbParams = { host: ip, port, dbname, user, password };
      const result = await embedWatermark(dbParams, table, primaryKey, column, message);
      
      setEmbedResult(`${result.message}。ID文件已自动下载，请妥善保存用于提取水印。`);
      showToast('水印嵌入成功！ID文件已下载', 'success');
      
      if (result.file_id) {
        setFileId(result.file_id);
      }
      
      if (result.downloadWarning) {
        showToast(result.downloadWarning, 'warning');
      }
    } catch (error) {
      setEmbedResult(`错误: ${error.message}`);
      showToast(`嵌入失败：${error.message}`, 'error');
    } finally {
      setIsEmbedding(false);
    }
  };

  // 提取水印
  const handleExtract = async () => {
    if (!connected || !table || !column || !primaryKey || !selectedFile) return;
    
    setIsExtracting(true);
    setExtractResult('');
    
    try {
      const dbParams = { host: ip, port, dbname, user, password };
      const result = await extractWatermarkWithFile(dbParams, table, primaryKey, column, selectedFile);
      
      if (result.success) {
        setExtractResult(`提取成功：${result.message} (恢复 ${result.recovered}/${result.blocks} 个区块)`);
        showToast(`水印提取成功！恢复 ${result.recovered}/${result.blocks} 个区块`, 'success');
      } else {
        setExtractResult(`提取失败：${result.error}`);
        showToast(`提取失败：${result.error}`, 'error');
      }
    } catch (error) {
      setExtractResult(`错误: ${error.message}`);
      showToast(`提取失败：${error.message}`, 'error');
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
      showToast('ID文件选择成功', 'success');
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

  // 返回上一步
  const goBack = () => {
    setCurrentStep(1);
    setConnected(false);
    setStatusMsg('未连接');
    setStepCompleted(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white py-8">
      <div className="container mx-auto max-w-md px-4">
        {/* Toast 组件 */}
        {toasts.map(toast => (
          <Toast
            key={toast.id}
            message={toast.message}
            type={toast.type}
            isVisible={toast.isVisible}
            onClose={() => removeToast(toast.id)}
            position="bottom"
          />
        ))}

        {/* 步骤指示器 */}
        <div className="mb-8 flex flex-col items-center">
          <div className="flex items-center space-x-4 mb-4">
            {/* Step 1 */}
            <div className="flex flex-col items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full text-sm font-medium transition-all duration-300 ease-in-out ${
                stepCompleted
                  ? 'bg-gradient-to-r from-green-400 to-emerald-500 text-white animate-bounce-subtle' 
                  : currentStep === 1
                  ? 'bg-gradient-to-r from-teal-400 to-green-400 text-white animate-pulse-slow'
                  : 'bg-gray-200 text-gray-600'
              }`}>
                {stepCompleted ? (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  '1'
                )}
              </div>
              <span className={`mt-2 text-xs font-medium transition-colors duration-150 ${
                currentStep === 1 ? 'text-teal-600' : stepCompleted ? 'text-green-600' : 'text-gray-500'
              }`}>
                数据库连接
              </span>
            </div>

            {/* 连接线 */}
            <div className="relative w-16 h-0.5 bg-gray-300 rounded-full overflow-hidden">
              <div className={`absolute top-0 left-0 h-full bg-gradient-to-r from-teal-400 to-green-400 rounded-full transition-all duration-300 ease-in-out ${
                currentStep >= 2 ? 'w-full animate-fill-line' : 'w-0'
              }`}></div>
            </div>

            {/* Step 2 */}
            <div className="flex flex-col items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full text-sm font-medium transition-all duration-300 ease-in-out ${
                currentStep >= 2 
                  ? 'bg-gradient-to-r from-teal-400 to-green-400 text-white animate-pulse-slow' 
                  : 'bg-gray-200 text-gray-600'
              }`}>
                2
              </div>
              <span className={`mt-2 text-xs font-medium transition-colors duration-150 ${
                currentStep >= 2 ? 'text-teal-600' : 'text-gray-500'
              }`}>
                水印管理
              </span>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          {/* Step 1: 数据库连接 */}
          {currentStep === 1 && (
            <div className="backdrop-blur-lg bg-white/70 p-6 rounded-2xl shadow-lg hover:shadow-xl hover:-translate-y-1 transition-all duration-150 ease-in-out animate-slide-in-left">
              <div className="text-center mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-teal-400 to-green-400 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 1.79 4 4 4h8c2.21 0 4-1.79 4-4V7c0-2.21-1.79-4-4-4H8c-2.21 0-4 1.79-4 4z" />
                  </svg>
                </div>
                <h2 className="text-xl font-semibold text-gray-900 mb-2">连接数据库</h2>
                <p className="text-sm text-gray-600">请输入 PGVector 数据库连接信息</p>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">主机地址</label>
                  <input
                    type="text"
                    value={ip}
                    onChange={e => {
                      setIp(e.target.value);
                      if (formErrors.ip) setFormErrors(prev => ({ ...prev, ip: null }));
                    }}
                    className={`w-full px-3 py-2 border rounded-lg transition-all duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-teal-300 ${
                      formErrors.ip ? 'border-red-300 focus:border-red-400' : 'border-gray-300 focus:border-teal-400'
                    }`}
                    disabled={loadingConn}
                    placeholder="localhost"
                  />
                  {formErrors.ip && (
                    <p className="mt-1 text-sm text-red-600 flex items-center animate-scale-in">
                      <svg className="w-4 h-4 mr-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {formErrors.ip}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">端口</label>
                  <input
                    type="number"
                    value={port}
                    onChange={e => {
                      setPort(parseInt(e.target.value, 10) || 0);
                      if (formErrors.port) setFormErrors(prev => ({ ...prev, port: null }));
                    }}
                    className={`w-full px-3 py-2 border rounded-lg transition-all duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-teal-300 ${
                      formErrors.port ? 'border-red-300 focus:border-red-400' : 'border-gray-300 focus:border-teal-400'
                    }`}
                    disabled={loadingConn}
                    placeholder="5432"
                  />
                  {formErrors.port && (
                    <p className="mt-1 text-sm text-red-600 flex items-center animate-scale-in">
                      <svg className="w-4 h-4 mr-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {formErrors.port}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">数据库名称</label>
                  <input
                    type="text"
                    value={dbname}
                    onChange={e => {
                      setDbname(e.target.value);
                      if (formErrors.dbname) setFormErrors(prev => ({ ...prev, dbname: null }));
                    }}
                    className={`w-full px-3 py-2 border rounded-lg transition-all duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-teal-300 ${
                      formErrors.dbname ? 'border-red-300 focus:border-red-400' : 'border-gray-300 focus:border-teal-400'
                    }`}
                    disabled={loadingConn}
                    placeholder="test"
                  />
                  {formErrors.dbname && (
                    <p className="mt-1 text-sm text-red-600 flex items-center animate-scale-in">
                      <svg className="w-4 h-4 mr-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {formErrors.dbname}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">用户名</label>
                  <input
                    type="text"
                    value={user}
                    onChange={e => {
                      setUser(e.target.value);
                      if (formErrors.user) setFormErrors(prev => ({ ...prev, user: null }));
                    }}
                    className={`w-full px-3 py-2 border rounded-lg transition-all duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-teal-300 ${
                      formErrors.user ? 'border-red-300 focus:border-red-400' : 'border-gray-300 focus:border-teal-400'
                    }`}
                    disabled={loadingConn}
                    placeholder="postgres"
                  />
                  {formErrors.user && (
                    <p className="mt-1 text-sm text-red-600 flex items-center animate-scale-in">
                      <svg className="w-4 h-4 mr-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {formErrors.user}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">密码</label>
                  <input
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-teal-400 focus:ring-2 focus:ring-teal-300 transition-all duration-150 ease-in-out"
                    disabled={loadingConn}
                    placeholder="输入数据库密码"
                  />
                </div>

                <button
                  onClick={handleConnect}
                  disabled={loadingConn}
                  className="w-full bg-gradient-to-r from-teal-400 to-green-400 hover:from-teal-500 hover:to-green-500 text-white font-medium py-3 rounded-lg hover:scale-105 transition-all duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none shadow-lg hover:shadow-xl"
                  style={{borderRadius: '0.5rem'}}
                >
                  {loadingConn ? (
                    <div className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-3 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      连接中...
                    </div>
                  ) : (
                    '连接数据库'
                  )}
                </button>

                {/* 连接状态 */}
                {statusMsg !== '未连接' && (
                  <div className={`p-3 rounded-lg transition-all duration-150 ease-in-out animate-scale-in ${
                    connected 
                      ? 'bg-green-50 border border-green-200' 
                      : 'bg-red-50 border border-red-200'
                  }`}>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${
                        connected ? 'bg-green-500 animate-pulse-slow' : 'bg-red-500'
                      }`}></div>
                      <span className={`text-sm ${
                        connected ? 'text-green-700' : 'text-red-700'
                      }`}>
                        {statusMsg}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Step 2: 水印操作 */}
          {currentStep === 2 && (
            <div className="space-y-6 animate-slide-in-right">
              {/* 返回按钮 */}
              <button
                onClick={goBack}
                className="flex items-center text-sm text-gray-600 hover:text-gray-800 transition-colors duration-150 ease-in-out"
              >
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                返回连接设置
              </button>

              {/* 表和列选择 */}
              <div className="backdrop-blur-lg bg-white/70 p-6 rounded-2xl shadow-lg hover:shadow-xl hover:-translate-y-1 transition-all duration-150 ease-in-out">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <svg className="w-5 h-5 text-teal-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 1.79 4 4 4h8c2.21 0 4-1.79 4-4V7c0-2.21-1.79-4-4-4H8c-2.21 0-4 1.79-4 4z" />
                  </svg>
                  数据库配置
                </h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">选择数据表</label>
                    <Combobox
                      options={tables}
                      value={table}
                      onChange={setTable}
                      placeholder="搜索并选择数据表"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">选择主键列</label>
                    <Combobox
                      options={primaryKeys}
                      value={primaryKey}
                      onChange={setPrimaryKey}
                      placeholder="搜索并选择主键列"
                      error={primaryKeys.length === 0 ? "该表无主键" : null}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">选择向量列</label>
                    <Combobox
                      options={columns}
                      value={column}
                      onChange={setColumn}
                      placeholder="搜索并选择向量列"
                    />
                  </div>
                </div>
              </div>

              {/* Tab 切换和操作 */}
              <div className="backdrop-blur-lg bg-white/70 p-6 rounded-2xl shadow-lg hover:shadow-xl hover:-translate-y-1 transition-all duration-150 ease-in-out">
                {/* Pills 切换 */}
                <div className="flex bg-gray-100 p-1 rounded-xl mb-6">
                  <button
                    onClick={() => setActiveTab('embed')}
                    className={`flex-1 flex items-center justify-center py-2 px-4 text-sm font-medium rounded-lg transition-all duration-150 ease-in-out ${
                      activeTab === 'embed'
                        ? 'bg-gradient-to-r from-teal-400 to-green-400 text-white shadow-sm'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-200'
                    }`}
                  >
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                    嵌入水印
                  </button>
                  <button
                    onClick={() => setActiveTab('extract')}
                    className={`flex-1 flex items-center justify-center py-2 px-4 text-sm font-medium rounded-lg transition-all duration-150 ease-in-out ${
                      activeTab === 'extract'
                        ? 'bg-gradient-to-r from-teal-400 to-green-400 text-white shadow-sm'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-200'
                    }`}
                  >
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    提取水印
                  </button>
                </div>

                {/* 嵌入水印 Tab */}
                {activeTab === 'embed' && (
                  <div className="space-y-4 animate-fade-in">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">水印消息 (32字符)</label>
                      <textarea
                        rows={3}
                        value={message}
                        onChange={e => setMessage(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-teal-400 focus:ring-2 focus:ring-teal-300 transition-all duration-150 ease-in-out resize-none"
                        disabled={!connected || !table || !column}
                        maxLength={32}
                        placeholder="输入32个字符的水印消息"
                      />
                      <div className="mt-1 flex justify-between items-center text-xs">
                        <span className={`transition-colors duration-150 ${
                          message.length === 32 ? 'text-teal-600 font-medium' : 'text-gray-500'
                        }`}>
                          {message.length}/32 字符
                        </span>
                        {message.length !== 32 && message.length > 0 && (
                          <span className="text-amber-600 flex items-center animate-scale-in">
                            <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                            </svg>
                            需要恰好32个字符
                          </span>
                        )}
                      </div>
                    </div>

                    <button
                      onClick={handleEmbed}
                      disabled={!connected || !table || !primaryKey || !column || !message || message.length !== 32 || isEmbedding}
                      className="w-full bg-gradient-to-r from-teal-400 to-green-400 hover:from-teal-500 hover:to-green-500 text-white font-medium py-3 rounded-lg hover:scale-105 transition-all duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none shadow-lg hover:shadow-xl"
                      style={{borderRadius: '0.5rem'}}
                    >
                      {isEmbedding ? (
                        <div className="flex items-center justify-center">
                          <svg className="animate-spin -ml-1 mr-3 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          嵌入中...
                        </div>
                      ) : (
                        <div className="flex items-center justify-center">
                          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l4-4m-4 4V4" />
                          </svg>
                          嵌入水印并下载ID文件
                        </div>
                      )}
                    </button>
                    
                    {embedResult && (
                      <div className="p-4 bg-green-50 border border-green-200 rounded-lg animate-scale-in">
                        <div className="flex items-start">
                          <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <div>
                            <h4 className="font-medium text-green-800 mb-1">嵌入结果</h4>
                            <p className="text-green-700 text-sm whitespace-pre-line">{embedResult}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 提取水印 Tab */}
                {activeTab === 'extract' && (
                  <div className="space-y-4 animate-fade-in">
                    {/* 文件上传区域 */}
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-all duration-150 ease-in-out">
                      <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      <div className="mt-4">
                        <label htmlFor="file-upload" className="cursor-pointer">
                          <span className="mt-2 block text-sm font-medium text-gray-700">
                            {fileName ? fileName : '点击选择ID文件'}
                          </span>
                          <span className="mt-1 block text-xs text-gray-500">
                            支持JSON格式文件
                          </span>
                        </label>
                        <input
                          id="file-upload"
                          ref={fileInputRef}
                          type="file"
                          className="sr-only"
                          onChange={handleFileChange}
                          accept=".json"
                        />
                      </div>
                      {selectedFile && (
                        <button
                          onClick={clearFileSelection}
                          className="mt-2 text-red-600 hover:text-red-800 text-sm font-medium transition-colors duration-150 ease-in-out"
                        >
                          清除选择
                        </button>
                      )}
                    </div>
                    
                    <div className="flex space-x-3">
                      <button
                        onClick={handleExtract}
                        disabled={!connected || !table || !primaryKey || !column || isExtracting || !selectedFile}
                        className="flex-1 bg-gradient-to-r from-teal-400 to-green-400 hover:from-teal-500 hover:to-green-500 text-white font-medium py-3 rounded-lg hover:scale-105 transition-all duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none shadow-lg hover:shadow-xl"
                        style={{borderRadius: '0.5rem'}}
                      >
                        {isExtracting ? (
                          <div className="flex items-center justify-center">
                            <svg className="animate-spin -ml-1 mr-3 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            提取中...
                          </div>
                        ) : (
                          <div className="flex items-center justify-center">
                            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l4-4m-4 4V4" />
                            </svg>
                            提取水印
                          </div>
                        )}
                      </button>
                      <button
                        onClick={clearFileSelection}
                        disabled={!selectedFile}
                        className="px-4 py-3 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 hover:border-gray-400 transition-all duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed"
                        style={{borderRadius: '0.5rem'}}
                      >
                        清除
                      </button>
                    </div>

                    {extractResult && (
                      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg animate-scale-in">
                        <div className="flex items-start">
                          <svg className="w-5 h-5 text-blue-500 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <div>
                            <h4 className="font-medium text-blue-800 mb-1">提取结果</h4>
                            <p className="text-blue-700 text-sm">{extractResult}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}