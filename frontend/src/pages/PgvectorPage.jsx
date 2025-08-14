import React, { useState, useEffect, useRef } from 'react';
import { 
  connectDB, 
  fetchTables, 
  fetchColumns, 
  fetchPrimaryKeys, 
  getVectorDimension, 
  checkModel, 
  trainModel, 
  getTrainingStatus,
  embedWatermark,
  extractWatermark,
  getVectorVisualization,
  getVectorVisualizationAsync,
  getVisualizationStatus
} from '../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';

// 现代化Toast组件
const ModernToast = ({ toast, onRemove }) => {
  useEffect(() => {
    const timer = setTimeout(() => onRemove(toast.id), 5000);
    return () => clearTimeout(timer);
  }, [toast.id, onRemove]);

  const getToastStyles = () => {
    switch (toast.type) {
      case 'success':
        return 'bg-green-500 border-green-400';
      case 'error':
        return 'bg-red-500 border-red-400';
      case 'warning':
        return 'bg-yellow-500 border-yellow-400';
      default:
        return 'bg-blue-500 border-blue-400';
    }
  };

  return (
    <div className={`${getToastStyles()} text-white px-6 py-4 rounded-xl shadow-lg border-l-4 transform transition-all duration-300 hover:scale-105`}>
      <div className="flex items-center justify-between">
        <span className="font-medium">{toast.message}</span>
        <button
          onClick={() => onRemove(toast.id)}
          className="ml-4 text-white hover:text-gray-200 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
};

// 步骤指示器组件
const StepIndicator = ({ currentStep, steps }) => {
  return (
    <div className="flex items-center justify-between mb-8">
      {steps.map((step, index) => (
        <div key={index} className="flex items-center">
          <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all duration-300 ${
            index + 1 <= currentStep 
              ? 'bg-gradient-to-r from-teal-500 to-cyan-500 border-teal-500 text-white' 
              : 'border-gray-300 text-gray-400'
          }`}>
            {index + 1 <= currentStep ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <span className="text-sm font-semibold">{index + 1}</span>
            )}
          </div>
          <span className={`ml-3 text-sm font-medium ${
            index + 1 <= currentStep ? 'text-teal-600' : 'text-gray-400'
          }`}>
            {step}
          </span>
          {index < steps.length - 1 && (
            <div className={`mx-4 h-0.5 w-16 ${
              index + 1 < currentStep ? 'bg-teal-500' : 'bg-gray-300'
            }`} />
          )}
        </div>
      ))}
    </div>
  );
};

// 现代化输入组件
const ModernInput = ({ label, type = 'text', value, onChange, placeholder, error, icon, ...props }) => {
  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">{label}</label>
      <div className="relative">
        {icon && (
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <div className="text-gray-400">{icon}</div>
          </div>
        )}
        <input
          type={type}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          className={`w-full ${icon ? 'pl-10' : 'pl-4'} pr-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all duration-200 ${
            error ? 'border-red-500 ring-2 ring-red-200' : ''
          }`}
          {...props}
        />
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
    </div>
  );
};

// 现代化按钮组件
const ModernButton = ({ children, variant = 'primary', size = 'md', loading = false, disabled = false, onClick, className = '', ...props }) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  const variants = {
    primary: 'bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-600 hover:to-cyan-600 text-white focus:ring-teal-500',
    secondary: 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50 focus:ring-gray-500',
    danger: 'bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white focus:ring-red-500',
    success: 'bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white focus:ring-green-500'
  };
  
  const sizes = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-6 py-3 text-base',
    lg: 'px-8 py-4 text-lg'
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={`${baseClasses} ${variants[variant]} ${sizes[size]} ${
        disabled || loading ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105 hover:shadow-lg'
      } ${className}`}
      {...props}
    >
      {loading && (
        <svg className="animate-spin -ml-1 mr-3 h-5 w-5" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      )}
      {children}
    </button>
  );
};

// 现代化卡片组件
const ModernCard = ({ children, className = '', title, subtitle, ...props }) => {
  return (
    <div className={`bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden ${className}`} {...props}>
      {(title || subtitle) && (
        <div className="px-6 py-4 border-b border-gray-100">
          {title && <h3 className="text-lg font-semibold text-gray-900">{title}</h3>}
          {subtitle && <p className="text-sm text-gray-600 mt-1">{subtitle}</p>}
        </div>
      )}
      <div className="p-6">
        {children}
      </div>
    </div>
  );
};

// 侧边栏导航组件
const Sidebar = ({ activeSection, onSectionChange, connected, modelExists }) => {
  const sections = [
    { id: 'connection', name: '数据库连接', icon: '🔗', enabled: true },
    { id: 'model', name: '模型管理', icon: '🤖', enabled: connected },
    { id: 'watermark', name: '水印操作', icon: '💧', enabled: connected && modelExists },
    { id: 'visualization', name: '数据可视化', icon: '📊', enabled: connected }
  ];

  return (
    <div className="w-64 bg-white border-r border-gray-200 h-screen sticky top-0">
      <div className="p-6">
        <div className="flex items-center space-x-3 mb-8">
          <div className="w-10 h-10 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 1.79 4 4 4h8c2.21 0 4-1.79 4-4V7c0-2.21-1.79-4-4-4H8c-2.21 0-4 1.79-4 4z" />
            </svg>
          </div>
          <div>
            <h2 className="text-lg font-bold text-gray-900">PGVector</h2>
            <p className="text-xs text-gray-500">向量数据库</p>
          </div>
        </div>

        <nav className="space-y-2">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => section.enabled && onSectionChange(section.id)}
              disabled={!section.enabled}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-left transition-all duration-200 ${
                activeSection === section.id
                  ? 'bg-gradient-to-r from-teal-50 to-cyan-50 text-teal-700 border border-teal-200'
                  : section.enabled
                  ? 'text-gray-700 hover:bg-gray-50'
                  : 'text-gray-400 cursor-not-allowed'
              }`}
            >
              <span className="text-lg">{section.icon}</span>
              <span className="font-medium">{section.name}</span>
              {!section.enabled && (
                <svg className="w-4 h-4 ml-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              )}
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
};

export default function PgvectorPage() {
  // 状态管理
  const [activeSection, setActiveSection] = useState('connection');
  const [connected, setConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  
  // 数据库连接参数
  const [connectionData, setConnectionData] = useState({
    ip: '',
    port: '',
    dbName: '',
    user: '',
    password: ''
  });
  const [errors, setErrors] = useState({});
  
  // 表和列选择
  const [tables, setTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState('');
  const [columns, setColumns] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState('');
  const [primaryKeys, setPrimaryKeys] = useState([]);
  const [selectedPrimaryKey, setSelectedPrimaryKey] = useState('');
  
  // 模型状态
  const [vectorDimension, setVectorDimension] = useState(null);
  const [modelExists, setModelExists] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingResult, setTrainingResult] = useState(null);
  
  // 训练参数
  const [trainingParams, setTrainingParams] = useState({
    epochs: 100,
    learningRate: 0.003,
    batchSize: 8196,
    valRatio: 0.15
  });
  
  // 水印嵌入操作
  const [watermarkData, setWatermarkData] = useState({
    message: '',
    encryptionKey: '',
    embedRate: 0.1,
    nonce: ''
  });
  const [keyFile, setKeyFile] = useState(null);
  
  // 水印提取操作（分离的状态）
  const [extractData, setExtractData] = useState({
    encryptionKey: '',
    embedRate: 0.1,
    nonce: ''
  });
  const [extractKeyFile, setExtractKeyFile] = useState(null);
  const [nonceFile, setNonceFile] = useState(null);
  const [extractNonceFile, setExtractNonceFile] = useState(null);
  const [lastNonce, setLastNonce] = useState('');
  const [lastEmbedRate, setLastEmbedRate] = useState(''); // 记录上次成功嵌入时使用的嵌入率
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [embedResult, setEmbedResult] = useState('');
  const [extractResult, setExtractResult] = useState('');
  
  // 可视化
  const [visualizationData, setVisualizationData] = useState(null);
  const [isVisualizing, setIsVisualizing] = useState(false);
  
  // Toast通知
  const [toasts, setToasts] = useState([]);

  // Toast管理
  const showToast = (message, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
  };

  const removeToast = (id) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  // 数据库连接
  const handleConnect = async () => {
    const newErrors = {};
    Object.keys(connectionData).forEach(key => {
      if (!connectionData[key].trim()) {
        newErrors[key] = '此字段不能为空';
      }
    });
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    try {
      setIsConnecting(true);
      const dbParams = {
        host: connectionData.ip,
        port: connectionData.port,
        dbname: connectionData.dbName,
        user: connectionData.user,
        password: connectionData.password
      };
      
      const response = await connectDB(dbParams);
      
      if (response.success) {
        setConnected(true);
        showToast('数据库连接成功！', 'success');
        
        // 获取表列表
        const tables = await fetchTables(dbParams);
        setTables(tables);
        
        setActiveSection('model');
      } else {
        showToast(`连接失败: ${response.message}`, 'error');
      }
    } catch (error) {
      showToast(`连接错误: ${error.message}`, 'error');
    } finally {
      setIsConnecting(false);
    }
  };

  // 获取列信息
  const handleTableSelect = async (tableName) => {
    setSelectedTable(tableName);
    try {
      const dbParams = {
        host: connectionData.ip,
        port: connectionData.port,
        dbname: connectionData.dbName,
        user: connectionData.user,
        password: connectionData.password
      };
      
      const [columns, primaryKeys] = await Promise.all([
        fetchColumns(dbParams, tableName),
        fetchPrimaryKeys(dbParams, tableName)
      ]);
      
      setColumns(columns);
      setPrimaryKeys(primaryKeys);
    } catch (error) {
      showToast(`获取表信息失败: ${error.message}`, 'error');
    }
  };

  // 检查模型
  const checkModelStatus = async () => {
    if (!selectedTable || !selectedColumn) {
      // 如果没有选择表和列，重置模型状态
      setVectorDimension(null);
      setModelExists(false);
      return;
    }
    
    try {
      const dbParams = {
        host: connectionData.ip,
        port: connectionData.port,
        dbname: connectionData.dbName,
        user: connectionData.user,
        password: connectionData.password
      };
      
      const dimResponse = await getVectorDimension(dbParams, selectedTable, selectedColumn);
      if (dimResponse.dimension) {
        setVectorDimension(dimResponse.dimension);
        
        const modelResponse = await checkModel(dimResponse.dimension);
        setModelExists(modelResponse.exists);
        
        // 只在模型状态发生变化时显示提示
        // 不显示toast，避免频繁弹出提示
      } else {
        // 如果无法获取维度，重置状态
        setVectorDimension(null);
        setModelExists(false);
      }
    } catch (error) {
      console.error('检查模型失败:', error);
      setVectorDimension(null);
      setModelExists(false);
    }
  };

  // 训练模型
  const handleTrainModel = async () => {
    if (!selectedTable || !selectedColumn || !vectorDimension) {
      showToast('请先选择表和列，并确保获取到向量维度', 'error');
      return;
    }
    
    try {
      setIsTraining(true);
      setTrainingProgress(0);
      
      const dbParams = {
        host: connectionData.ip,
        port: connectionData.port,
        dbname: connectionData.dbName,
        user: connectionData.user,
        password: connectionData.password
      };
      
      // 转换字段名以匹配后端API
      const apiTrainingParams = {
        epochs: trainingParams.epochs,
        learning_rate: trainingParams.learningRate,
        batch_size: trainingParams.batchSize,
        val_ratio: trainingParams.valRatio
      };
      
      const response = await trainModel(dbParams, selectedTable, selectedColumn, vectorDimension, apiTrainingParams);
      
      if (response.task_id) {
        showToast('模型训练已开始', 'success');
        
        // 轮询训练状态
        const pollTrainingStatus = async () => {
          try {
            const status = await getTrainingStatus(response.task_id);
            
            if (status.progress !== undefined) {
              setTrainingProgress(status.progress);
            }
            
            if (status.status === 'completed') {
              setIsTraining(false);
              setTrainingProgress(100);
              setTrainingResult(status); // 保存完整的训练结果
              await checkModelStatus();
              
              // 显示详细的训练结果
              const metrics = status.final_metrics || {};
              const performanceMsg = `训练完成！\n最佳BER: ${(status.best_ber * 100).toFixed(2)}%\n性能等级: ${status.performance_level || 'N/A'}\n最终训练损失: ${metrics.train_loss?.toFixed(4) || 'N/A'}\n最终验证损失: ${metrics.val_loss?.toFixed(4) || 'N/A'}`;
              showToast(performanceMsg, 'success');
            } else if (status.status === 'failed') {
              setIsTraining(false);
              showToast(`训练失败: ${status.error || '未知错误'}`, 'error');
            } else if (status.status === 'running' || status.status === 'starting') {
              // 继续轮询
              setTimeout(pollTrainingStatus, 2000); // 每2秒检查一次
            }
          } catch (error) {
            console.error('获取训练状态失败:', error);
            // 如果获取状态失败，继续尝试
            setTimeout(pollTrainingStatus, 3000);
          }
        };
        
        // 开始轮询
        setTimeout(pollTrainingStatus, 1000); // 1秒后开始第一次检查
      } else {
        setIsTraining(false);
        showToast(`训练失败: ${response.message}`, 'error');
      }
    } catch (error) {
      setIsTraining(false);
      showToast(`训练错误: ${error.message}`, 'error');
    }
  };

  // 嵌入水印
  const handleEmbedWatermark = async () => {
    if (!watermarkData.message || watermarkData.message.length !== 16) {
      showToast('请输入16个字符的水印信息', 'error');
      return;
    }
    if (!watermarkData.encryptionKey || !selectedPrimaryKey) {
      showToast('请填写加密密钥，并选择主键列', 'error');
      return;
    }
    
    try {
      setIsEmbedding(true);
      const dbParams = {
        host: connectionData.ip,
        port: connectionData.port,
        dbname: connectionData.dbName,
        user: connectionData.user,
        password: connectionData.password
      };
      
      const response = await embedWatermark(
        dbParams,
        selectedTable,
        selectedPrimaryKey,
        selectedColumn,
        watermarkData.message,
        watermarkData.embedRate,
        watermarkData.encryptionKey
      );
      
      if (response.success) {
        setLastEmbedRate(watermarkData.embedRate);
        
        // 保存返回的nonce
        if (response.nonce) {
          setLastNonce(response.nonce);
        }
        
        setEmbedResult(`${response.message}\n\n💡 提示：提取水印时请使用相同的嵌入率 ${(watermarkData.embedRate * 100).toFixed(1)}% 和相同的解密密钥以确保正确提取。\n\n⚠️ 重要：请保存以下nonce值，提取水印时需要：\n${response.nonce}`);
        showToast(`水印嵌入成功！使用了 ${(watermarkData.embedRate * 100).toFixed(1)}% 的嵌入率`, 'success');
      } else {
        setEmbedResult(`错误: ${response.error || "未知错误"}`);
        showToast(`嵌入失败: ${response.error || response.message || "未知错误"}`, 'error');
      }
    } catch (error) {
      showToast(`嵌入错误: ${error.message}`, 'error');
    } finally {
      setIsEmbedding(false);
    }
  };

  // 提取水印
  const handleExtractWatermark = async () => {
    if (!extractData.encryptionKey || !extractData.nonce || !selectedPrimaryKey) {
      showToast('请填写加密密钥、nonce值，并选择主键列', 'error');
      return;
    }
    
    // 检查嵌入率是否与上次嵌入时一致
    if (lastEmbedRate && extractData.embedRate !== lastEmbedRate) {
      const confirmExtract = window.confirm(
        `检测到当前嵌入率 (${extractData.embedRate}) 与上次嵌入时的嵌入率 (${lastEmbedRate}) 不一致。\n\n为了正确提取水印，建议使用相同的嵌入率。\n\n是否继续提取？`
      );
      if (!confirmExtract) {
        return;
      }
    }
    
    try {
      setIsExtracting(true);
      const dbParams = {
        host: connectionData.ip,
        port: connectionData.port,
        dbname: connectionData.dbName,
        user: connectionData.user,
        password: connectionData.password
      };
      
      const response = await extractWatermark(
        dbParams,
        selectedTable,
        selectedPrimaryKey,
        selectedColumn,
        extractData.embedRate,
        extractData.encryptionKey,
        extractData.nonce
      );
      
      if (response.success) {
        setExtractResult(response.message);
        showToast('水印提取成功！', 'success');
      } else {
        showToast(`提取失败: ${response.message}`, 'error');
      }
    } catch (error) {
      showToast(`提取错误: ${error.message}`, 'error');
    } finally {
      setIsExtracting(false);
    }
  };

  // 生成可视化
  const handleVisualize = async () => {
    if (!selectedTable || !selectedColumn) {
      showToast('请先选择表和列', 'error');
      return;
    }
    
    try {
      setIsVisualizing(true);
      // 暂时生成模拟数据用于演示
      const mockData = Array.from({ length: 100 }, (_, i) => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        type: i % 2 === 0 ? '原始向量' : '水印向量'
      }));
      
      setVisualizationData(mockData);
      showToast('可视化生成成功！（演示数据）', 'success');
    } catch (error) {
      showToast(`可视化错误: ${error.message}`, 'error');
    } finally {
      setIsVisualizing(false);
    }
  };

  // 监听表和列的变化
  useEffect(() => {
    if (selectedTable && selectedColumn) {
      // 使用setTimeout进行简单的防抖，避免快速切换时的重复调用
      const timer = setTimeout(() => {
        checkModelStatus();
      }, 300);
      
      return () => clearTimeout(timer);
    } else {
      // 当表或列未选择时，重置相关状态
      setVectorDimension(null);
      setModelExists(false);
    }
  }, [selectedTable, selectedColumn]);

  // 监听列选择变化，重置主键选择
  useEffect(() => {
    if (!selectedColumn) {
      setSelectedPrimaryKey('');
    }
  }, [selectedColumn]);

  // 渲染连接部分
  const renderConnectionSection = () => (
    <ModernCard>
      <div className="text-center mb-6">
        <div className="w-16 h-16 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 1.79 4 4 4h8c2.21 0 4-1.79 4-4V7c0-2.21-1.79-4-4-4H8c-2.21 0-4 1.79-4 4z" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">连接PGVector数据库</h2>
        <p className="text-gray-600">基于PostgreSQL向量扩展的水印嵌入系统</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ModernInput
          label="IP地址"
          value={connectionData.ip}
          onChange={(e) => setConnectionData(prev => ({ ...prev, ip: e.target.value }))}
          placeholder="localhost"
          error={errors.ip}
          icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9v-9m0-9v9" />
          </svg>}
        />
        <ModernInput
          label="端口"
          value={connectionData.port}
          onChange={(e) => setConnectionData(prev => ({ ...prev, port: e.target.value }))}
          placeholder="5432"
          error={errors.port}
          icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z" />
          </svg>}
        />
        <ModernInput
          label="数据库名"
          value={connectionData.dbName}
          onChange={(e) => setConnectionData(prev => ({ ...prev, dbName: e.target.value }))}
          placeholder="database_name"
          error={errors.dbName}
          icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 1.79 4 4 4h8c2.21 0 4-1.79 4-4V7c0-2.21-1.79-4-4-4H8c-2.21 0-4 1.79-4 4z" />
          </svg>}
        />
        <ModernInput
          label="用户名"
          value={connectionData.user}
          onChange={(e) => setConnectionData(prev => ({ ...prev, user: e.target.value }))}
          placeholder="username"
          error={errors.user}
          icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>}
        />
        <div className="md:col-span-2">
          <ModernInput
            label="密码"
            type="password"
            value={connectionData.password}
            onChange={(e) => setConnectionData(prev => ({ ...prev, password: e.target.value }))}
            placeholder="password"
            error={errors.password}
            icon={<svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>}
          />
        </div>
      </div>
      
      <div className="mt-6 flex justify-end">
        <ModernButton 
          onClick={handleConnect} 
          disabled={connected || isConnecting}
          loading={isConnecting}
        >
          {connected ? '已连接' : isConnecting ? '连接中...' : '连接数据库'}
        </ModernButton>
      </div>
      
      {connected && (
        <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-xl">
          <div className="flex items-center">
            <svg className="w-5 h-5 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            <span className="text-green-700 font-medium">数据库连接成功</span>
          </div>
        </div>
      )}
    </ModernCard>
  );

  // 渲染模型管理部分
  const renderModelSection = () => (
    <ModernCard>
      <div className="text-center mb-6">
        <div className="w-16 h-16 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">模型管理</h2>
        <p className="text-gray-600">训练和管理水印模型</p>
      </div>

      <div className="space-y-6">
        {/* 表和列选择 */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">数据表</label>
            <select
              value={selectedTable}
              onChange={(e) => handleTableSelect(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all duration-200"
            >
              <option value="">选择表</option>
              {tables.map(table => (
                <option key={table} value={table}>{table}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">向量列</label>
            <select
              value={selectedColumn}
              onChange={(e) => setSelectedColumn(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all duration-200"
              disabled={!selectedTable}
            >
              <option value="">选择列</option>
              {columns.map(column => (
                <option key={column} value={column}>{column}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">主键</label>
            <select
              value={selectedPrimaryKey}
              onChange={(e) => setSelectedPrimaryKey(e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all duration-200"
              disabled={!selectedTable}
            >
              <option value="">选择主键</option>
              {primaryKeys.map(key => (
                <option key={key} value={key}>{key}</option>
              ))}
            </select>
          </div>
        </div>
        
        {/* 向量维度显示 */}
        {vectorDimension && (
          <div className="p-4 rounded-xl border bg-teal-50 border-teal-200 text-teal-800">
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full mr-3 bg-teal-500"></div>
              <span className="font-medium">向量维度: {vectorDimension}</span>
            </div>
          </div>
        )}

        {/* 模型状态显示 */}
        <div className={`p-4 rounded-xl border ${
          modelExists 
            ? 'bg-green-50 border-green-200 text-green-800' 
            : 'bg-yellow-50 border-yellow-200 text-yellow-800'
        }`}>
          <div className="flex items-center">
            <div className={`w-3 h-3 rounded-full mr-3 ${
              modelExists ? 'bg-green-500' : 'bg-yellow-500'
            }`}></div>
            <span className="font-medium">
              {modelExists ? '模型已存在' : '需要训练模型'}
            </span>
          </div>
        </div>
          
        {/* 训练参数 */}
        {!modelExists && selectedTable && selectedColumn && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">训练参数</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">训练轮数</label>
                <ModernInput
                  type="number"
                  value={trainingParams.epochs}
                  onChange={(e) => setTrainingParams(prev => ({ ...prev, epochs: parseInt(e.target.value) || 100 }))}
                  min="1"
                  max="1000"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">学习率</label>
                <ModernInput
                  type="number"
                  step="0.0001"
                  value={trainingParams.learningRate}
                  onChange={(e) => setTrainingParams(prev => ({ ...prev, learningRate: parseFloat(e.target.value) || 0.001 }))}
                  min="0.0001"
                  max="0.1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">批次大小</label>
                <ModernInput
                  type="number"
                  value={trainingParams.batchSize}
                  onChange={(e) => setTrainingParams(prev => ({ ...prev, batchSize: parseInt(e.target.value) || 32 }))}
                  min="1"
                  max="512"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">验证比例</label>
                <ModernInput
                  type="number"
                  step="0.1"
                  value={trainingParams.valRatio}
                  onChange={(e) => setTrainingParams(prev => ({ ...prev, valRatio: parseFloat(e.target.value) || 0.2 }))}
                  min="0.1"
                  max="0.5"
                />
              </div>
            </div>
            
            <ModernButton
              onClick={handleTrainModel}
              loading={isTraining}
              disabled={isTraining}
              className="w-full"
            >
              {isTraining ? '训练中...' : '开始训练模型'}
            </ModernButton>
            
            {isTraining && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-gray-600">
                  <span>训练进度</span>
                  <span>{trainingProgress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                   <div 
                     className="bg-gradient-to-r from-teal-500 to-cyan-500 h-2 rounded-full transition-all duration-300"
                     style={{ width: `${trainingProgress}%` }}
                   ></div>
                 </div>
              </div>
            )}
            
            {/* 训练结果 */}
            {trainingResult && (
              <div className="p-6 rounded-xl border bg-green-50 border-green-200">
                <div className="flex items-center mb-4">
                  <div className="w-4 h-4 rounded-full mr-3 bg-green-500"></div>
                  <span className="font-semibold text-green-800 text-lg">训练完成</span>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-white p-3 rounded-lg border border-green-200">
                    <div className="text-sm text-gray-600">最佳BER</div>
                    <div className="text-lg font-bold text-green-700">
                      {trainingResult.best_ber ? `${(trainingResult.best_ber * 100).toFixed(3)}%` : 'N/A'}
                    </div>
                  </div>
                  <div className="bg-white p-3 rounded-lg border border-green-200">
                    <div className="text-sm text-gray-600">性能等级</div>
                    <div className="text-lg font-bold text-green-700">
                      {trainingResult.performance_level || 'N/A'}
                    </div>
                  </div>
                </div>
                
                {trainingResult.final_metrics && (
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="bg-white p-3 rounded-lg border border-green-200">
                      <div className="text-sm text-gray-600">最终训练损失</div>
                      <div className="text-lg font-bold text-blue-700">
                        {trainingResult.final_metrics.train_loss?.toFixed(4) || 'N/A'}
                      </div>
                    </div>
                    <div className="bg-white p-3 rounded-lg border border-green-200">
                      <div className="text-sm text-gray-600">最终验证损失</div>
                      <div className="text-lg font-bold text-blue-700">
                        {trainingResult.final_metrics.val_loss?.toFixed(4) || 'N/A'}
                      </div>
                    </div>
                  </div>
                )}
                
                {trainingResult.train_params && (
                  <div className="bg-white p-3 rounded-lg border border-green-200">
                    <div className="text-sm text-gray-600 mb-2">训练参数</div>
                    <div className="text-sm text-gray-700">
                      Epochs: {trainingResult.train_params.epochs} | 
                      LR: {trainingResult.train_params.learning_rate} | 
                      Batch: {trainingResult.train_params.batch_size} | 
                      Val Ratio: {trainingResult.train_params.val_ratio}
                    </div>
                  </div>
                )}
                
                {trainingResult.suggestions && trainingResult.suggestions.length > 0 && (
                  <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <div className="text-sm font-medium text-yellow-800 mb-2">优化建议</div>
                    <ul className="text-sm text-yellow-700 space-y-1">
                      {trainingResult.suggestions.map((suggestion, index) => (
                        <li key={index}>• {suggestion}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </ModernCard>
  );

  // 渲染水印操作部分
  const renderWatermarkSection = () => (
    <div className="space-y-8">
      {/* 水印操作概览 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-blue-50 to-indigo-100 p-6 rounded-2xl border border-blue-200">
          <div className="flex items-center mb-4">
            <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center mr-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-blue-900">水印嵌入</h3>
              <p className="text-blue-700 text-sm">将加密信息嵌入向量数据</p>
            </div>
          </div>
          <div className="text-sm text-blue-800 space-y-1">
            <p>• 支持自定义消息内容</p>
            <p>• AES-GCM 加密保护</p>
            <p>• 可调节嵌入率</p>
          </div>
        </div>

        <div className="bg-gradient-to-br from-teal-50 to-emerald-100 p-6 rounded-2xl border border-teal-200">
          <div className="flex items-center mb-4">
            <div className="w-12 h-12 bg-teal-500 rounded-xl flex items-center justify-center mr-4">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-teal-900">水印提取</h3>
              <p className="text-teal-700 text-sm">从向量数据中提取水印信息</p>
            </div>
          </div>
          <div className="text-sm text-teal-800 space-y-1">
            <p>• 需要正确的解密密钥</p>
            <p>• 匹配嵌入时的参数</p>
            <p>• 验证数据完整性</p>
          </div>
        </div>
      </div>

      <ModernCard title="水印嵌入" subtitle="在向量数据中嵌入水印信息" className="border-l-4 border-l-blue-500">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              水印信息 <span className="text-red-500">*</span>
              <span className="text-xs text-gray-500 ml-2">(限制16个字符)</span>
            </label>
            <input
              type="text"
              value={watermarkData.message}
              onChange={(e) => {
                const value = e.target.value;
                if (value.length <= 16) {
                  setWatermarkData(prev => ({ ...prev, message: value }));
                }
              }}
              maxLength={16}
              className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all duration-200 ${
                watermarkData.message.length === 16 ? 'border-green-300 bg-green-50' : 
                watermarkData.message.length > 12 ? 'border-yellow-300 bg-yellow-50' : 
                'border-gray-300'
              }`}
              placeholder="输入要嵌入的水印信息（16字符）"
            />
            <div className="flex justify-between items-center mt-1">
              <div className="text-xs text-gray-500">
                {watermarkData.message.length === 0 && '请输入16个字符的水印信息'}
                {watermarkData.message.length > 0 && watermarkData.message.length < 16 && '还需要输入更多字符'}
                {watermarkData.message.length === 16 && '✓ 字符数量正确'}
              </div>
              <div className={`text-xs font-medium ${
                watermarkData.message.length === 16 ? 'text-green-600' : 
                watermarkData.message.length > 12 ? 'text-yellow-600' : 
                'text-gray-500'
              }`}>
                {watermarkData.message.length}/16
              </div>
            </div>
          </div>
          
          {/* 密钥输入区域 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">AES-GCM 加密密钥</label>
            <div className="space-y-3">
              {/* 手动输入密钥 */}
              {!keyFile && (
                <input
                  type="password"
                  value={watermarkData.encryptionKey}
                  onChange={(e) => setWatermarkData(prev => ({ ...prev, encryptionKey: e.target.value }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all duration-200"
                  placeholder="输入AES-GCM加密密钥"
                />
              )}

              {/* 文件上传 */}
              {keyFile === null && (
                <div className="border-2 border-dashed border-gray-300 rounded-xl hover:border-teal-400 transition-colors duration-200">
                  <input
                    type="file"
                    id="keyFileInput"
                    className="hidden"
                    accept=".key,.txt"
                    onChange={(e) => {
                      const file = e.target.files[0];
                      if (file) {
                        setKeyFile(file);
                        const reader = new FileReader();
                        reader.onload = (event) => {
                          setWatermarkData(prev => ({ ...prev, encryptionKey: event.target.result }));
                        };
                        reader.readAsText(file);
                      }
                    }}
                  />
                  <label
                    htmlFor="keyFileInput"
                    className="flex flex-col items-center justify-center py-6 cursor-pointer"
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
                <div className="flex items-center justify-between p-3 bg-teal-50 border border-teal-200 rounded-xl">
                  <div className="flex items-center">
                    <svg className="w-4 h-4 text-teal-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <span className="text-sm text-teal-700">{keyFile.name}</span>
                  </div>
                  <button
                    onClick={() => {
                      setKeyFile(null);
                      setWatermarkData(prev => ({ ...prev, encryptionKey: '' }));
                    }}
                    className="text-red-500 hover:text-red-700 transition-colors duration-200"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              )}

              <div className="text-xs text-gray-500 bg-blue-50 p-3 rounded-xl border border-blue-200">
                💡 <strong>密钥说明：</strong>系统将使用AES-GCM算法对水印信息进行加密，生成密文和验证标签用于水印嵌入。请确保密钥的安全性。
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ModernInput
              label="嵌入率"
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={watermarkData.embedRate}
              onChange={(e) => setWatermarkData(prev => ({ ...prev, embedRate: parseFloat(e.target.value) }))}
            />
            <ModernInput
              label="随机数种子 (可选)"
              value={watermarkData.nonce}
              onChange={(e) => setWatermarkData(prev => ({ ...prev, nonce: e.target.value }))}
              placeholder="留空将自动生成"
            />
          </div>
          
          <ModernButton
            onClick={handleEmbedWatermark}
            loading={isEmbedding}
            disabled={isEmbedding || !modelExists}
            className="w-full"
          >
            {isEmbedding ? '嵌入中...' : '嵌入水印'}
          </ModernButton>
          
          {embedResult && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-xl">
              <div className="flex items-start">
                <svg className="w-5 h-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div className="flex-1">
                  <h4 className="font-medium text-green-800 mb-1">嵌入结果</h4>
                  <p className="text-green-700 text-sm whitespace-pre-line">{embedResult}</p>
                  
                  {/* 添加复制nonce的按钮 */}
                  {lastNonce && (
                    <div className="mt-4 flex gap-2">
                      <ModernButton
                        onClick={() => {
                          navigator.clipboard.writeText(lastNonce);
                          showToast('已复制nonce到剪贴板', 'success');
                        }}
                        size="sm"
                        variant="secondary"
                        className="text-xs"
                      >
                        <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 003-3v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2v12" />
                        </svg>
                        复制nonce
                      </ModernButton>
                      
                      <ModernButton
                        onClick={() => {
                          const element = document.createElement('a');
                          const file = new Blob([lastNonce], {type: 'text/plain'});
                          element.href = URL.createObjectURL(file);
                          element.download = `pgvector_watermark_nonce_${new Date().toISOString().slice(0,10)}.txt`;
                          document.body.appendChild(element);
                          element.click();
                          document.body.removeChild(element);
                          showToast('已下载nonce文件', 'success');
                        }}
                        size="sm"
                        variant="secondary"
                        className="text-xs"
                      >
                        <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        下载nonce
                      </ModernButton>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </ModernCard>

      <ModernCard title="水印提取" subtitle="从向量数据中提取水印信息" className="border-l-4 border-l-teal-500">
        <div className="space-y-4">
          {/* 嵌入率输入 */}
          <ModernInput
            label="嵌入率"
            type="number"
            step="0.01"
            min="0.01"
            max="1"
            value={extractData.embedRate}
            onChange={(e) => setExtractData(prev => ({ ...prev, embedRate: parseFloat(e.target.value) }))}
            placeholder="0.1"
          />

          {/* 密钥输入区域（提取时需要） */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">AES-GCM 解密密钥</label>
            <div className="space-y-3">
              {/* 手动输入密钥 */}
              {!extractKeyFile && (
                <input
                  type="password"
                  value={extractData.encryptionKey}
                  onChange={(e) => setExtractData(prev => ({ ...prev, encryptionKey: e.target.value }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all duration-200"
                  placeholder="输入用于解密的AES-GCM密钥"
                />
              )}

              {/* 文件上传 */}
              {extractKeyFile === null && (
                <div className="border-2 border-dashed border-gray-300 rounded-xl hover:border-teal-400 transition-colors duration-200">
                  <input
                    type="file"
                    id="extractKeyFileInput"
                    className="hidden"
                    accept=".key,.txt"
                    onChange={(e) => {
                      const file = e.target.files[0];
                      if (file) {
                        setExtractKeyFile(file);
                        const reader = new FileReader();
                        reader.onload = (event) => {
                          setExtractData(prev => ({ ...prev, encryptionKey: event.target.result }));
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
              {extractKeyFile && (
                <div className="flex items-center justify-between p-3 bg-teal-50 border border-teal-200 rounded-xl">
                  <div className="flex items-center">
                    <svg className="w-4 h-4 text-teal-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <span className="text-sm text-teal-700">{extractKeyFile.name}</span>
                  </div>
                  <button
                    onClick={() => {
                      setExtractKeyFile(null);
                      setExtractData(prev => ({ ...prev, encryptionKey: '' }));
                    }}
                    className="text-red-500 hover:text-red-700 transition-colors duration-200"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              )}

              <div className="text-xs text-gray-500 bg-amber-50 p-3 rounded-xl border border-amber-200">
                ⚠️ <strong>注意：</strong>解密密钥必须与嵌入时使用的密钥完全一致，才能正确提取明文消息
              </div>
            </div>
          </div>
          
          {/* Nonce输入区域 */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">随机数种子 (Nonce)</label>
            <div className="space-y-3">
              {/* 手动输入nonce */}
              {!extractNonceFile && (
                <input
                  type="text"
                  value={extractData.nonce}
                  onChange={(e) => setExtractData(prev => ({ ...prev, nonce: e.target.value }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all duration-200"
                  placeholder="输入嵌入时使用的随机数种子"
                />
              )}

              {/* 使用上次嵌入的nonce */}
              {lastNonce && !extractNonceFile && (
                <ModernButton
                  onClick={() => {
                    setExtractData(prev => ({ ...prev, nonce: lastNonce }));
                    showToast('已使用上次嵌入的nonce', 'success');
                  }}
                  size="sm"
                  variant="secondary"
                  className="text-xs"
                >
                  <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  使用上次嵌入的nonce
                </ModernButton>
              )}

              {/* nonce文件上传 */}
              {extractNonceFile === null && (
                <div className="border-2 border-dashed border-gray-300 rounded-xl hover:border-teal-400 transition-colors duration-200">
                  <input
                    type="file"
                    id="extractNonceFileInput"
                    className="hidden"
                    accept=".txt"
                    onChange={(e) => {
                      const file = e.target.files[0];
                      if (file) {
                        setExtractNonceFile(file);
                        const reader = new FileReader();
                        reader.onload = (event) => {
                          setExtractData(prev => ({ ...prev, nonce: event.target.result.trim() }));
                        };
                        reader.readAsText(file);
                      }
                    }}
                  />
                  <label
                    htmlFor="extractNonceFileInput"
                    className="flex flex-col items-center justify-center py-4 cursor-pointer"
                  >
                    <svg className="w-8 h-8 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <p className="text-sm text-gray-600 text-center">
                      点击选择nonce文件<br />
                      <span className="text-xs text-gray-500">支持 .txt 格式</span>
                    </p>
                  </label>
                </div>
              )}

              {/* nonce文件信息显示 */}
              {extractNonceFile && (
                <div className="flex items-center justify-between p-3 bg-teal-50 border border-teal-200 rounded-xl">
                  <div className="flex items-center">
                    <svg className="w-4 h-4 text-teal-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <span className="text-sm text-teal-700">{extractNonceFile.name}</span>
                  </div>
                  <button
                    onClick={() => {
                      setExtractNonceFile(null);
                      setExtractData(prev => ({ ...prev, nonce: '' }));
                    }}
                    className="text-red-500 hover:text-red-700 transition-colors duration-200"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              )}

              <div className="text-xs text-gray-500 bg-amber-50 p-3 rounded-xl border border-amber-200">
                ⚠️ <strong>注意：</strong>nonce必须与嵌入时使用的完全一致，才能正确提取水印
              </div>
            </div>
          </div>
          
          <ModernButton
            onClick={handleExtractWatermark}
            loading={isExtracting}
            disabled={isExtracting || !modelExists}
            variant="secondary"
            className="w-full"
          >
            {isExtracting ? '提取中...' : '提取水印'}
          </ModernButton>
          
          {extractResult && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-xl">
              <p className="text-blue-700">
                <span className="font-medium">提取结果:</span> {extractResult}
              </p>
            </div>
          )}
        </div>
      </ModernCard>
    </div>
  );

  // 渲染可视化部分
  const renderVisualizationSection = () => (
    <ModernCard>
      {/* 数据可视化标题和图标 */}
      <div className="text-center mb-6">
        <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">数据可视化</h2>
        <p className="text-gray-600">生成向量数据的可视化图表</p>
      </div>

      <div className="space-y-6">
        <ModernButton
          onClick={handleVisualize}
          loading={isVisualizing}
          disabled={isVisualizing || !selectedTable || !selectedColumn}
          className="w-full"
        >
          {isVisualizing ? '生成中...' : '生成可视化'}
        </ModernButton>
        
        {visualizationData && (
          <div className="h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={visualizationData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" />
                <YAxis dataKey="y" />
                <Tooltip />
                <Scatter dataKey="y" fill="#14b8a6" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </ModernCard>
  );

  // 渲染主要内容
  const renderMainContent = () => {
    switch (activeSection) {
      case 'connection':
        return renderConnectionSection();
      case 'model':
        return renderModelSection();
      case 'watermark':
        return renderWatermarkSection();
      case 'visualization':
        return renderVisualizationSection();
      default:
        return renderConnectionSection();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-teal-50 pt-20">
      <div className="flex min-h-screen">
        <Sidebar
          activeSection={activeSection}
          onSectionChange={setActiveSection}
          connected={connected}
          modelExists={modelExists}
        />
        
        {/* 主内容区域 */}
        <div className="flex-1 overflow-hidden">
          <div className="h-full overflow-y-auto">
            <div className="p-6 lg:p-8 xl:p-10">
              {/* 页面标题区域 */}
              <div className="mb-8">
                <div className="flex items-center justify-between">
                  <div>
                    <h1 className="text-3xl font-bold text-gray-900 mb-2">
                      PgVector 数据库水印系统
                    </h1>
                    <p className="text-gray-600 text-lg">
                      基于向量数据库的智能水印嵌入与提取平台
                    </p>
                  </div>
                  
                  {/* 状态指示器 */}
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span className="text-sm font-medium text-gray-700">
                        {connected ? '已连接' : '未连接'}
                      </span>
                    </div>
                    {connected && (
                      <div className="flex items-center space-x-2">
                        <div className={`w-3 h-3 rounded-full ${modelExists ? 'bg-blue-500' : 'bg-yellow-500'}`}></div>
                        <span className="text-sm font-medium text-gray-700">
                          {modelExists ? '模型就绪' : '模型未训练'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* 主要内容容器 */}
              <div className="max-w-6xl mx-auto">
                <div className="bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-6 lg:p-8">
                  {renderMainContent()}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Toast通知 */}
      <div className="fixed top-24 right-6 space-y-4 z-50">
        {toasts.map(toast => (
          <ModernToast key={toast.id} toast={toast} onRemove={removeToast} />
        ))}
      </div>
    </div>
  );
}


