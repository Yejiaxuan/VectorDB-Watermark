import React, { useState, useEffect } from 'react';
import Toast from '../components/Toast';
import Combobox from '../components/Combobox';
import {
  connectDB,
  fetchTables,
  fetchColumns,
  fetchPrimaryKeys,
  embedWatermark,
  extractWatermark,
  getVectorDimension,
  checkModel,
  trainModel,
  getTrainingStatus
} from '../api';

export default function PgvectorPage() {
  // —— 步骤控制 —__
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

  // —— 表单验证 —__
  const [formErrors, setFormErrors] = useState({});

  // —— 表/列 列表 ——  
  const [tables, setTables] = useState([]);
  const [table, setTable] = useState('');
  const [primaryKeys, setPrimaryKeys] = useState([]);
  const [primaryKey, setPrimaryKey] = useState('');
  const [columns, setColumns] = useState([]);
  const [column, setColumn] = useState('');

  // —— 向量维度和模型状态 —__
  const [vectorDimension, setVectorDimension] = useState(null);
  const [modelExists, setModelExists] = useState(false);
  const [modelChecking, setModelChecking] = useState(false);
  const [modelPath, setModelPath] = useState('');

  // —— 训练相关状态 —__
  const [trainingTaskId, setTrainingTaskId] = useState('');
  const [trainingStatus, setTrainingStatus] = useState('');
  const [trainingMessage, setTrainingMessage] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(100);
  const [trainingMetrics, setTrainingMetrics] = useState({
    train_loss: 0,
    train_ber: 1,
    val_loss: 0,
    val_ber: 1
  });
  const [finalResults, setFinalResults] = useState(null);
  const [trainingParams, setTrainingParams] = useState({
    epochs: 100,
    learning_rate: 0.0003,
    batch_size: 8192,
    val_ratio: 0.15
  });
  const [showTrainingParams, setShowTrainingParams] = useState(false);

  // —— Tab 控制 —__
  const [activeTab, setActiveTab] = useState('embed');

  // —— 水印操作状态 ——  
  const [message, setMessage] = useState('ABCDEFGHIJKLMNOP'); // 16字节明文
  const [encryptionKey, setEncryptionKey] = useState(''); // 加密密钥
  const [keyFile, setKeyFile] = useState(null); // 密钥文件
  const [embedRate, setEmbedRate] = useState(0.1); // 默认10%嵌入率
  const [lastEmbedRate, setLastEmbedRate] = useState(null); // 记录上次成功嵌入时使用的嵌入率
  const [embedResult, setEmbedResult] = useState('');
  const [extractResult, setExtractResult] = useState('');
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [nonce, setNonce] = useState(''); // 新增：nonce字段
  const [lastNonce, setLastNonce] = useState(''); // 新增：记录上次嵌入返回的nonce

  // —— Toast 相关 —__
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

  // 检查向量维度和模型
  const checkVectorDimensionAndModel = async () => {
    if (!connected || !table || !column) return;
    
    setModelChecking(true);
    try {
      // 获取向量维度
      const dimResult = await getVectorDimension({ host: ip, port, dbname, user, password }, table, column);
      const dimension = dimResult.dimension;
      setVectorDimension(dimension);
      
      // 检查模型是否存在
      const modelResult = await checkModel(dimension);
      setModelExists(modelResult.exists);
      setModelPath(modelResult.model_path);
      
      if (modelResult.exists) {
        showToast(`检测到 ${dimension} 维向量，对应模型已存在`, 'success');
      } else {
        showToast(`检测到 ${dimension} 维向量，需要训练对应模型`, 'warning');
      }
    } catch (err) {
      showToast(`检查失败：${err.message}`, 'error');
      setVectorDimension(null);
      setModelExists(false);
    } finally {
      setModelChecking(false);
    }
  };

  // 启动训练
  const handleTrainModel = async () => {
    if (!connected || !table || !column || !vectorDimension) return;
    
    try {
      const result = await trainModel(
        { host: ip, port, dbname, user, password }, 
        table, 
        column, 
        vectorDimension,
        trainingParams
      );
      setTrainingTaskId(result.task_id);
      setIsTraining(true);
      setTrainingStatus('starting');
      setTrainingMessage(result.message);
      setTrainingProgress(0);
      setCurrentEpoch(0);
      setTotalEpochs(trainingParams.epochs);
      setFinalResults(null);
      showToast('训练任务已启动', 'success');
      
      // 开始轮询训练状态
      pollTrainingStatus(result.task_id);
    } catch (err) {
      showToast(`启动训练失败：${err.message}`, 'error');
    }
  };

  // 轮询训练状态
  const pollTrainingStatus = async (taskId) => {
    const interval = setInterval(async () => {
      try {
        const status = await getTrainingStatus(taskId);
        setTrainingStatus(status.status);
        setTrainingMessage(status.message || '');
        setTrainingProgress(status.progress || 0);
        setCurrentEpoch(status.current_epoch || 0);
        setTotalEpochs(status.total_epochs || trainingParams.epochs);
        
        if (status.metrics) {
          setTrainingMetrics(status.metrics);
        }
        
        if (status.status === 'completed') {
          setIsTraining(false);
          setModelExists(true);
          setFinalResults({
            best_ber: status.best_ber,
            performance_level: status.performance_level,
            suggestions: status.suggestions || [],
            final_metrics: status.final_metrics,
            train_params: status.train_params
          });
          
          // 根据性能水平显示不同的消息
          const performanceMessages = {
            excellent: '🎉 训练效果极佳！',
            good: '👍 训练效果良好！', 
            poor: '⚠️ 训练效果不佳，建议调整参数重新训练'
          };
          
          showToast(
            `${performanceMessages[status.performance_level]} ${status.message}`, 
            status.performance_level === 'poor' ? 'warning' : 'success'
          );
          clearInterval(interval);
        } else if (status.status === 'failed') {
          setIsTraining(false);
          showToast(`训练失败：${status.error}`, 'error');
          clearInterval(interval);
        }
      } catch (err) {
        console.error('获取训练状态失败:', err);
        clearInterval(interval);
      }
    }, 2000); // 每2秒轮询一次
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
      setVectorDimension(null); setModelExists(false);
    }
  }, [connected]);

  // 当表格或列变更时，重置水印状态
  useEffect(() => {
    setEmbedResult('');
    setExtractResult('');
    setVectorDimension(null);
    setModelExists(false);
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

  // 当选择列后自动检查维度和模型
  useEffect(() => {
    if (column) {
      checkVectorDimensionAndModel();
    }
  }, [column]);

  // 嵌入水印
  const handleEmbed = async () => {
    if (!connected || !message || !table || !column || !primaryKey || message.length !== 16 || !encryptionKey) return;
    
    setIsEmbedding(true);
    setEmbedResult('');
    setExtractResult('');
    
    try {
      const dbParams = { host: ip, port, dbname, user, password };
      const result = await embedWatermark(dbParams, table, primaryKey, column, message, embedRate, encryptionKey);
      
      // 保存返回的nonce
      if (result.nonce) {
        setLastNonce(result.nonce);
      }
      
      setEmbedResult(`${result.message}\n\n💡 提示：提取水印时请使用相同的嵌入率 ${(embedRate * 100).toFixed(1)}% 和相同的解密密钥以确保正确提取。\n\n⚠️ 重要：请保存以下nonce值，提取水印时需要：\n${result.nonce}`);
      setLastEmbedRate(embedRate);
      showToast(`水印嵌入成功！使用了 ${(embedRate * 100).toFixed(1)}% 的嵌入率`, 'success');
      
    } catch (error) {
      setEmbedResult(`错误: ${error.message}`);
      showToast(`嵌入失败：${error.message}`, 'error');
    } finally {
      setIsEmbedding(false);
    }
  };

  // 提取水印
  const handleExtract = async () => {
    if (!connected || !table || !column || !primaryKey || !encryptionKey || !nonce) return;
    
    setIsExtracting(true);
    setExtractResult('');
    
    try {
      const dbParams = { host: ip, port, dbname, user, password };
      const result = await extractWatermark(dbParams, table, primaryKey, column, embedRate, encryptionKey, nonce);
      
      if (result.success) {
        const stats = result.stats ? ` (有效解码: ${result.valid_decodes}/${result.total_decodes})` : '';
        setExtractResult(`提取成功：${result.message} (恢复 ${result.recovered}/${result.blocks} 个区块)${stats}`);
        showToast('水印提取成功！', 'success');
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

              {/* 向量维度和模型状态 */}
              {vectorDimension && (
                <div className="backdrop-blur-lg bg-white/70 p-6 rounded-2xl shadow-lg hover:shadow-xl hover:-translate-y-1 transition-all duration-150 ease-in-out">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <svg className="w-5 h-5 text-purple-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    模型状态
                  </h3>
                  
                  <div className="space-y-4">
                    {/* 向量维度显示 */}
                    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center">
                        <svg className="w-5 h-5 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zM21 5a2 2 0 00-2-2h-4a2 2 0 00-2 2v12a4 4 0 004 4h4a4 4 0 001-4V5z" />
                        </svg>
                        <span className="text-sm font-medium text-gray-700">向量维度</span>
                      </div>
                      <span className="text-lg font-bold text-blue-600">{vectorDimension}维</span>
                    </div>

                    {/* 模型状态显示 */}
                    <div className={`p-4 rounded-lg border-2 ${
                      modelExists 
                        ? 'bg-green-50 border-green-200' 
                        : 'bg-orange-50 border-orange-200'
                    }`}>
                      <div className="flex items-start justify-between">
                        <div className="flex items-start">
                          <div className={`w-3 h-3 rounded-full mt-1 mr-3 ${
                            modelExists ? 'bg-green-500' : 'bg-orange-500'
                          }`}></div>
                          <div>
                            <h4 className={`font-medium ${
                              modelExists ? 'text-green-800' : 'text-orange-800'
                            }`}>
                              {modelExists ? '模型已就绪' : '需要训练模型'}
                            </h4>
                            <p className={`text-sm mt-1 ${
                              modelExists ? 'text-green-600' : 'text-orange-600'
                            }`}>
                              {modelExists 
                                ? `${vectorDimension}维向量的水印模型已存在，可以直接进行水印操作`
                                : `尚未检测到${vectorDimension}维向量的水印模型，需要先训练模型`
                              }
                            </p>
                            {modelExists && modelPath && (
                              <p className="text-xs text-gray-500 mt-1">
                                模型路径: {modelPath}
                              </p>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* 训练按钮和状态 */}
                      {!modelExists && (
                        <div className="mt-4">
                          {!isTraining ? (
                            <div className="space-y-4">
                              {/* 训练参数设置按钮 */}
                              <button
                                onClick={() => setShowTrainingParams(!showTrainingParams)}
                                className="w-full text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 transition-colors duration-150"
                              >
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center">
                                    <svg className="w-4 h-4 text-gray-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                    </svg>
                                    <span className="text-sm font-medium text-gray-700">训练参数设置</span>
                                  </div>
                                  <svg className={`w-4 h-4 text-gray-400 transition-transform duration-150 ${
                                    showTrainingParams ? 'rotate-180' : ''
                                  }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                  </svg>
                                </div>
                              </button>

                              {/* 训练参数表单 */}
                              {showTrainingParams && (
                                <div className="space-y-4 p-4 bg-gray-50 rounded-lg border animate-scale-in">
                                  <div className="grid grid-cols-2 gap-4">
                                    <div>
                                      <label className="block text-xs font-medium text-gray-700 mb-1">训练轮数</label>
                                      <input
                                        type="number"
                                        min="1"
                                        max="1000"
                                        value={trainingParams.epochs}
                                        onChange={e => setTrainingParams(prev => ({
                                          ...prev,
                                          epochs: parseInt(e.target.value) || 100
                                        }))}
                                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-orange-400"
                                      />
                                    </div>
                                    <div>
                                      <label className="block text-xs font-medium text-gray-700 mb-1">学习率</label>
                                      <input
                                        type="number"
                                        min="0.0001"
                                        max="0.01"
                                        step="0.0001"
                                        value={trainingParams.learning_rate}
                                        onChange={e => setTrainingParams(prev => ({
                                          ...prev,
                                          learning_rate: parseFloat(e.target.value) || 0.0003
                                        }))}
                                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-orange-400"
                                      />
                                    </div>
                                    <div>
                                      <label className="block text-xs font-medium text-gray-700 mb-1">批处理大小</label>
                                      <select
                                        value={trainingParams.batch_size}
                                        onChange={e => setTrainingParams(prev => ({
                                          ...prev,
                                          batch_size: parseInt(e.target.value)
                                        }))}
                                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-orange-400"
                                      >
                                        <option value={1024}>1024</option>
                                        <option value={2048}>2048</option>
                                        <option value={4096}>4096</option>
                                        <option value={8192}>8192</option>
                                        <option value={16384}>16384</option>
                                      </select>
                                    </div>
                                    <div>
                                      <label className="block text-xs font-medium text-gray-700 mb-1">验证集比例</label>
                                      <select
                                        value={trainingParams.val_ratio}
                                        onChange={e => setTrainingParams(prev => ({
                                          ...prev,
                                          val_ratio: parseFloat(e.target.value)
                                        }))}
                                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-orange-400"
                                      >
                                        <option value={0.1}>10%</option>
                                        <option value={0.15}>15%</option>
                                        <option value={0.2}>20%</option>
                                        <option value={0.25}>25%</option>
                                      </select>
                                    </div>
                                  </div>
                                  <div className="text-xs text-gray-500 bg-blue-50 p-2 rounded">
                                    💡 <strong>参数说明：</strong>训练轮数控制训练时长，学习率影响收敛速度，批处理大小影响内存使用和训练稳定性
                                  </div>
                                </div>
                              )}

                              {/* 开始训练按钮 */}
                              <button
                                onClick={handleTrainModel}
                                disabled={modelChecking || !vectorDimension}
                                className="w-full bg-gradient-to-r from-orange-400 to-red-400 hover:from-orange-500 hover:to-red-500 text-white font-medium py-3 px-4 rounded-lg hover:scale-105 transition-all duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none shadow-lg hover:shadow-xl"
                              >
                                <div className="flex items-center justify-center">
                                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                  </svg>
                                  开始训练模型
                                </div>
                              </button>
                            </div>
                          ) : (
                            <div className="space-y-4">
                              {/* 训练进度条 */}
                              <div className="space-y-2">
                                <div className="flex items-center justify-between text-sm">
                                  <span className="text-orange-700 font-medium">训练进度</span>
                                  <span className="text-orange-600">{trainingProgress}%</span>
                                </div>
                                <div className="w-full bg-orange-100 rounded-full h-2">
                                  <div 
                                    className="bg-gradient-to-r from-orange-400 to-red-400 h-2 rounded-full transition-all duration-300 ease-out"
                                    style={{ width: `${trainingProgress}%` }}
                                  ></div>
                                </div>
                                <div className="flex items-center justify-between text-xs text-gray-600">
                                  <span>Epoch {currentEpoch}/{totalEpochs}</span>
                                  <span>BER: {(trainingMetrics.val_ber * 100).toFixed(2)}%</span>
                                </div>
                              </div>

                              {/* 训练状态 */}
                              <div className="flex items-center">
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-orange-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                <span className="text-orange-700 font-medium">正在训练模型...</span>
                              </div>

                              {/* 训练消息 */}
                              {trainingMessage && (
                                <p className="text-sm text-orange-600 bg-orange-100 p-3 rounded">
                                  {trainingMessage}
                                </p>
                              )}

                              {/* 实时指标 */}
                              <div className="grid grid-cols-2 gap-2 text-xs">
                                <div className="bg-blue-50 p-2 rounded">
                                  <div className="text-blue-600 font-medium">训练损失</div>
                                  <div className="text-blue-800">{trainingMetrics.train_loss.toFixed(4)}</div>
                                </div>
                                <div className="bg-green-50 p-2 rounded">
                                  <div className="text-green-600 font-medium">验证BER</div>
                                  <div className="text-green-800">{(trainingMetrics.val_ber * 100).toFixed(2)}%</div>
                                </div>
                              </div>

                              {/* 任务ID */}
                              <div className="text-xs text-gray-500">
                                训练ID: {trainingTaskId}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* 训练结果展示 */}
                      {finalResults && (
                        <div className="mt-4 space-y-3">
                          <div className={`p-4 rounded-lg border-2 ${
                            finalResults.performance_level === 'excellent' ? 'bg-green-50 border-green-200' :
                            finalResults.performance_level === 'good' ? 'bg-blue-50 border-blue-200' :
                            'bg-red-50 border-red-200'
                          }`}>
                            <div className="flex items-start">
                              <div className={`w-3 h-3 rounded-full mt-1 mr-3 ${
                                finalResults.performance_level === 'excellent' ? 'bg-green-500' :
                                finalResults.performance_level === 'good' ? 'bg-blue-500' :
                                'bg-red-500'
                              }`}></div>
                              <div className="flex-1">
                                <h4 className={`font-medium ${
                                  finalResults.performance_level === 'excellent' ? 'text-green-800' :
                                  finalResults.performance_level === 'good' ? 'text-blue-800' :
                                  'text-red-800'
                                }`}>
                                  训练完成 - {
                                    finalResults.performance_level === 'excellent' ? '效果极佳' :
                                    finalResults.performance_level === 'good' ? '效果良好' :
                                    '效果不佳'
                                  }
                                </h4>
                                <div className="mt-2 text-sm space-y-1">
                                  <div>验证错误率：<span className="font-medium">{(finalResults.best_ber * 100).toFixed(2)}%</span></div>
                                  <div>训练参数：{finalResults.train_params.epochs}轮，学习率{finalResults.train_params.learning_rate}</div>
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* 建议信息 */}
                          {finalResults.suggestions && finalResults.suggestions.length > 0 && (
                            <div className="bg-yellow-50 border border-yellow-200 p-3 rounded-lg">
                              <h5 className="text-yellow-800 font-medium text-sm mb-2">💡 优化建议</h5>
                              <ul className="text-sm text-yellow-700 space-y-1">
                                {finalResults.suggestions.map((suggestion, index) => (
                                  <li key={index} className="flex items-start">
                                    <span className="mr-2">•</span>
                                    <span>{suggestion}</span>
                                  </li>
                                ))}
                              </ul>
                              <button
                                onClick={() => {
                                  setFinalResults(null);
                                  setModelExists(false);
                                  setShowTrainingParams(true);
                                }}
                                className="mt-3 text-xs bg-yellow-100 hover:bg-yellow-200 text-yellow-800 px-3 py-1 rounded transition-colors duration-150"
                              >
                                重新训练
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

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
                        <label className="block text-sm font-medium text-gray-700 mb-1">明文消息 (16字符)</label>
                        <textarea
                          rows={2}
                          value={message}
                          onChange={e => setMessage(e.target.value)}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-teal-400 focus:ring-2 focus:ring-teal-300 transition-all duration-150 ease-in-out resize-none"
                          disabled={!connected || !table || !column}
                          maxLength={16}
                          placeholder="输入16个字符的明文消息"
                        />
                        <div className="mt-1 flex justify-between items-center text-xs">
                          <span className={`transition-colors duration-150 ${
                            message.length === 16 ? 'text-teal-600 font-medium' : 'text-gray-500'
                          }`}>
                            {message.length}/16 字符
                          </span>
                          {message.length !== 16 && message.length > 0 && (
                            <span className="text-amber-600 flex items-center animate-scale-in">
                              <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.734 0l-7.92 13.5c-.77.833-.192 2.5 1.732 2.5z" />
                              </svg>
                              需要恰好16个字符
                            </span>
                          )}
                        </div>
                      </div>

                      {/* 加密密钥输入区域 */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">AES-GCM 加密密钥</label>
                        <div className="space-y-3">
                          

                          {/* 手动输入密钥 */}
                          {!keyFile && (
                            <input
                              type="password"
                              value={encryptionKey}
                              onChange={e => setEncryptionKey(e.target.value)}
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-teal-400 focus:ring-2 focus:ring-teal-300 transition-all duration-150 ease-in-out"
                              disabled={!connected || !table || !column}
                              placeholder="输入AES-GCM加密密钥"
                            />
                          )}

                          {/* 文件上传 */}
                          {keyFile === null && (
                            <div className="border-2 border-dashed border-gray-300 rounded-lg hover:border-teal-400 transition-colors duration-150">
                              <input
                                type="file"
                                id="keyFileInput"
                                className="hidden"
                                accept=".key,.txt"
                                onChange={e => {
                                  const file = e.target.files[0];
                                  if (file) {
                                    setKeyFile(file);
                                    const reader = new FileReader();
                                    reader.onload = (event) => {
                                      setEncryptionKey(event.target.result);
                                    };
                                    reader.readAsText(file);
                                  }
                                }}
                              />
                              <label
                                htmlFor="keyFileInput"
                                className="flex flex-col items-center justify-center py-4 cursor-pointer"
                              >
                                <svg className="w-8 h-8 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                                <p className="text-sm text-gray-600 text-center">
                                  点击选择密钥文件<br />
                                  <span className="text-xs text-gray-500">支持 .key, .txt 格式</span>
                                </p>
                              </label>
                            </div>
                          )}

                          {/* 文件信息显示 */}
                          {keyFile && (
                            <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
                              <div className="flex items-center">
                                <svg className="w-4 h-4 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                                <span className="text-sm text-blue-700">{keyFile.name}</span>
                              </div>
                              <button
                                onClick={() => {
                                  setKeyFile(null);
                                  setEncryptionKey('');
                                }}
                                className="text-red-500 hover:text-red-700 transition-colors duration-150"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                              </button>
                            </div>
                          )}

                          <div className="text-xs text-gray-500 bg-blue-50 p-2 rounded">
                            💡 <strong>密钥说明：</strong>系统将使用AES-GCM算法对16字节明文进行加密，生成16字节密文和16字节验证标签，总共32字节用于水印嵌入
                          </div>
                        </div>
                      </div>

                      {/* 水印嵌入率 */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">水印嵌入率</label>
                        <div className="space-y-2">
                          <input
                            type="range"
                            min="0.01"
                            max="1"
                            step="0.01"
                            value={embedRate}
                            onChange={e => setEmbedRate(parseFloat(e.target.value))}
                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                            disabled={!connected || !table || !column}
                          />
                          <div className="flex justify-between items-center text-xs">
                            <span className="text-gray-500">1%</span>
                            <span className="text-teal-600 font-medium bg-teal-50 px-2 py-1 rounded">
                              {(embedRate * 100).toFixed(1)}%
                            </span>
                            <span className="text-gray-500">100%</span>
                          </div>
                          <p className="text-xs text-gray-500">
                            控制用于嵌入水印的向量比例，推荐使用10-20%
                          </p>
                        </div>
                      </div>

                      {/* 操作按钮 */}
                      <button
                        onClick={handleEmbed}
                        disabled={!connected || !table || !primaryKey || !column || !message || message.length !== 16 || !encryptionKey || isEmbedding || !modelExists}
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
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                            </svg>
                            {!modelExists && vectorDimension ? '需要先训练模型' : (!encryptionKey ? '需要输入密钥' : '嵌入水印')}
                          </div>
                        )}
                      </button>

                      {/* 模型不存在提示 */}
                      {!modelExists && vectorDimension && (
                        <div className="p-3 bg-orange-50 border border-orange-200 rounded-lg animate-scale-in">
                          <div className="flex items-center">
                            <svg className="w-4 h-4 text-orange-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.734 0l-7.92 13.5c-.77.833-.192 2.5 1.732 2.5z" />
                            </svg>
                            <p className="text-sm text-orange-700">
                              请先训练 {vectorDimension} 维向量的水印模型，才能进行水印操作
                            </p>
                          </div>
                        </div>
                      )}
                      
                      {embedResult && (
                        <div className="p-4 bg-green-50 border border-green-200 rounded-lg animate-scale-in">
                          <div className="flex items-start">
                            <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            <div>
                              <h4 className="font-medium text-green-800 mb-1">嵌入结果</h4>
                              <p className="text-green-700 text-sm whitespace-pre-line">{embedResult}</p>
                              
                              {/* 添加复制nonce的按钮 */}
                              {lastNonce && (
                                <div className="mt-4">
                                  <button
                                    onClick={() => {
                                      navigator.clipboard.writeText(lastNonce);
                                      showToast('已复制nonce到剪贴板', 'success');
                                    }}
                                    className="bg-green-100 hover:bg-green-200 text-green-800 text-xs font-medium py-1 px-3 rounded transition-colors duration-150"
                                  >
                                    <span className="flex items-center">
                                      <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 003-3v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2v12" />
                                      </svg>
                                      复制nonce
                                    </span>
                                  </button>
                                  
                                  {/* 添加下载nonce的按钮 */}
                                  <button
                                    onClick={() => {
                                      const element = document.createElement('a');
                                      const file = new Blob([lastNonce], {type: 'text/plain'});
                                      element.href = URL.createObjectURL(file);
                                      element.download = `watermark_nonce_${new Date().toISOString().slice(0,10)}.txt`;
                                      document.body.appendChild(element);
                                      element.click();
                                      document.body.removeChild(element);
                                      showToast('已下载nonce文件', 'success');
                                    }}
                                    className="ml-2 bg-blue-100 hover:bg-blue-200 text-blue-800 text-xs font-medium py-1 px-3 rounded transition-colors duration-150"
                                  >
                                    <span className="flex items-center">
                                      <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                                      </svg>
                                      下载nonce
                                    </span>
                                  </button>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* 提取水印 Tab */}
                  {activeTab === 'extract' && (
                    <div className="space-y-4 animate-fade-in">
                      {/* 密钥输入区域（提取时需要） */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">AES-GCM 解密密钥</label>
                        <div className="space-y-3">
                          {/* 手动输入密钥 */}
                          {!keyFile && (
                            <input
                              type="password"
                              value={encryptionKey}
                              onChange={e => setEncryptionKey(e.target.value)}
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-teal-400 focus:ring-2 focus:ring-teal-300 transition-all duration-150 ease-in-out"
                              disabled={!connected || !table || !column}
                              placeholder="输入用于解密的AES-GCM密钥"
                            />
                          )}

                          {/* 文件上传 */}
                          {keyFile === null && (
                            <div className="border-2 border-dashed border-gray-300 rounded-lg hover:border-teal-400 transition-colors duration-150">
                              <input
                                type="file"
                                id="extractKeyFileInput"
                                className="hidden"
                                accept=".key,.txt"
                                onChange={e => {
                                  const file = e.target.files[0];
                                  if (file) {
                                    setKeyFile(file);
                                    const reader = new FileReader();
                                    reader.onload = (event) => {
                                      setEncryptionKey(event.target.result);
                                    };
                                    reader.readAsText(file);
                                  }
                                }}
                              />
                              <label
                                htmlFor="extractKeyFileInput"
                                className="flex flex-col items-center justify-center py-4 cursor-pointer"
                              >
                                <svg className="w-8 h-8 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                                <p className="text-sm text-gray-600 text-center">
                                  点击选择密钥文件<br />
                                  <span className="text-xs text-gray-500">支持 .key, .txt 格式</span>
                                </p>
                              </label>
                            </div>
                          )}

                          {/* 文件信息显示 */}
                          {keyFile && (
                            <div className="flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
                              <div className="flex items-center">
                                <svg className="w-4 h-4 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                                <span className="text-sm text-blue-700">{keyFile.name}</span>
                              </div>
                              <button
                                onClick={() => {
                                  setKeyFile(null);
                                  setEncryptionKey('');
                                }}
                                className="text-red-500 hover:text-red-700 transition-colors duration-150"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                              </button>
                            </div>
                          )}

                          <div className="text-xs text-gray-500 bg-amber-50 p-2 rounded border border-amber-200">
                            ⚠️ <strong>注意：</strong>解密密钥必须与嵌入时使用的密钥完全一致，才能正确提取明文消息
                          </div>
                        </div>
                      </div>

                      {/* nonce输入区域（提取时需要） */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          nonce <span className="text-red-600">*</span>
                        </label>
                        <div className="space-y-3">
                          <input
                            type="text"
                            value={nonce}
                            onChange={e => setNonce(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-teal-400 focus:ring-2 focus:ring-teal-300 transition-all duration-150 ease-in-out"
                            disabled={!connected || !table || !column}
                            placeholder="输入用于解密的nonce（嵌入水印时生成的十六进制字符串）"
                          />
                          <div className="text-xs text-gray-500 bg-amber-50 p-2 rounded border border-amber-200">
                            ⚠️ <strong>重要：</strong>必须提供嵌入水印时生成的nonce，否则无法正确解密
                          </div>
                          
                          {/* 上传nonce文件 */}
                          <div className="mt-2">
                            <label className="block text-xs text-gray-600 mb-1">或上传nonce文件：</label>
                            <input
                              type="file"
                              id="nonceFileInput"
                              className="hidden"
                              accept=".txt"
                              onChange={e => {
                                const file = e.target.files[0];
                                if (file) {
                                  const reader = new FileReader();
                                  reader.onload = (event) => {
                                    setNonce(event.target.result.trim());
                                    showToast('成功加载nonce文件', 'success');
                                  };
                                  reader.readAsText(file);
                                }
                              }}
                            />
                            <label
                              htmlFor="nonceFileInput"
                              className="flex items-center justify-center px-4 py-2 border border-gray-300 rounded-lg cursor-pointer bg-white hover:bg-gray-50 transition-colors duration-150"
                            >
                              <svg className="w-4 h-4 mr-2 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                              </svg>
                              <span className="text-sm text-gray-700">选择nonce文件</span>
                            </label>
                          </div>
                        </div>
                      </div>

                      {/* 水印嵌入率 */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">水印嵌入率</label>
                        <div className="space-y-2">
                          <input
                            type="range"
                            min="0.01"
                            max="1"
                            step="0.01"
                            value={embedRate}
                            onChange={e => setEmbedRate(parseFloat(e.target.value))}
                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                            disabled={!connected || !table || !column}
                          />
                          <div className="flex justify-between items-center text-xs">
                            <span className="text-gray-500">1%</span>
                            <span className="text-blue-600 font-medium bg-blue-50 px-2 py-1 rounded">
                              {(embedRate * 100).toFixed(1)}%
                            </span>
                            <span className="text-gray-500">100%</span>
                          </div>
                          <div className={`rounded-lg p-3 ${
                            lastEmbedRate !== null && Math.abs(embedRate - lastEmbedRate) > 0.001
                              ? 'bg-red-50 border border-red-200'
                              : 'bg-amber-50 border border-amber-200'
                          }`}>
                            <div className="flex items-start">
                              <svg className={`w-4 h-4 mr-2 mt-0.5 flex-shrink-0 ${
                                lastEmbedRate !== null && Math.abs(embedRate - lastEmbedRate) > 0.001
                                  ? 'text-red-500'
                                  : 'text-amber-500'
                              }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.734 0l-7.92 13.5c-.77.833-.192 2.5 1.732 2.5z" />
                            </svg>
                            <div>
                              <p className={`text-xs font-medium mb-1 ${
                                lastEmbedRate !== null && Math.abs(embedRate - lastEmbedRate) > 0.001

                                  ? 'text-red-700'
                                  : 'text-amber-700'
                              }`}>
                                {lastEmbedRate !== null && Math.abs(embedRate - lastEmbedRate) > 0.001
                                  ? '⚠️ 嵌入率不匹配'
                                  : '重要提示'
                                }
                              </p>
                              <p className={`text-xs ${
                                lastEmbedRate !== null && Math.abs(embedRate - lastEmbedRate) > 0.001
                                  ? 'text-red-600'
                                  : 'text-amber-600'
                              }`}>
                                {lastEmbedRate !== null 
                                  ? Math.abs(embedRate - lastEmbedRate) > 0.001
                                    ? `当前嵌入率 ${(embedRate * 100).toFixed(1)}% 与上次成功嵌入时的 ${(lastEmbedRate * 100).toFixed(1)}% 不一致，可能无法正确提取水印。`
                                    : `当前嵌入率 ${(embedRate * 100).toFixed(1)}% 与上次成功嵌入时保持一致，可以正确提取水印。`
                                  : '提取时的嵌入率应与嵌入水印时使用的嵌入率保持一致，才能正确提取水印信息。'
                                }
                              </p>
                              {lastEmbedRate !== null && Math.abs(embedRate - lastEmbedRate) > 0.001 && (
                                <button
                                  onClick={() => setEmbedRate(lastEmbedRate)}
                                  className="mt-2 text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition-colors duration-150"
                                >
                                  恢复到 {(lastEmbedRate * 100).toFixed(1)}%
                                </button>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                      </div>

                      {/* 操作按钮 */}
                      <button
                        onClick={handleExtract}
                        disabled={!connected || !table || !primaryKey || !column || !encryptionKey || !nonce || isExtracting || !modelExists}
                        className="w-full bg-gradient-to-r from-teal-400 to-green-400 hover:from-teal-500 hover:to-green-500 text-white font-medium py-3 rounded-lg hover:scale-105 transition-all duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none shadow-lg hover:shadow-xl"
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
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                            {!modelExists && vectorDimension ? '需要先训练模型' : (!encryptionKey ? '需要输入密钥' : !nonce ? '需要输入nonce' : '提取水印')}
                          </div>
                        )}
                      </button>

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
