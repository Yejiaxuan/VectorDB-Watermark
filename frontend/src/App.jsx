import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import PgvectorPage from './pages/PgvectorPage';
import MilvusPage from './pages/MilvusPage';

// 导航组件
function Navigation() {
  const location = useLocation();
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      isScrolled ? 'bg-white/95 backdrop-blur-md shadow-lg' : 'bg-white/80 backdrop-blur-sm'
    }`}>
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo和标题 */}
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-200">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                数据库水印系统
              </h1>
              <p className="text-xs text-gray-500">Database Watermarking System</p>
            </div>
          </Link>

          {/* 导航链接 */}
          <div className="hidden md:flex items-center space-x-1">
            <Link
              to="/"
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                location.pathname === '/' 
                  ? 'bg-blue-100 text-blue-700' 
                  : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
              }`}
            >
              首页
            </Link>
            <Link
              to="/pgvector"
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                location.pathname === '/pgvector' 
                  ? 'bg-teal-100 text-teal-700' 
                  : 'text-gray-600 hover:text-teal-600 hover:bg-teal-50'
              }`}
            >
              PGVector
            </Link>
            <Link
              to="/milvus"
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                location.pathname === '/milvus' 
                  ? 'bg-purple-100 text-purple-700' 
                  : 'text-gray-600 hover:text-purple-600 hover:bg-purple-50'
              }`}
            >
              Milvus
            </Link>
          </div>

          {/* 移动端菜单按钮 */}
          <button className="md:hidden p-2 rounded-lg hover:bg-gray-100">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </nav>
  );
}

// 首页组件
function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* 英雄区域 */}
      <div className="pt-32 pb-20">
        <div className="container mx-auto px-6 text-center">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-teal-600 bg-clip-text text-transparent">
                智能数据库
              </span>
              <br />
              <span className="text-gray-800">水印系统</span>
            </h1>
            <p className="text-xl text-gray-600 mb-12 leading-relaxed">
              基于深度学习的向量数据库水印技术，为您的数据提供安全可靠的版权保护
            </p>
            
            {/* 特性卡片 */}
            <div className="grid md:grid-cols-3 gap-6 mb-16">
              <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:shadow-xl transition-all duration-300">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mb-4 mx-auto">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold mb-2">安全可靠</h3>
                <p className="text-gray-600 text-sm">采用先进的加密算法，确保水印信息的安全性</p>
              </div>
              
              <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:shadow-xl transition-all duration-300">
                <div className="w-12 h-12 bg-gradient-to-br from-teal-500 to-teal-600 rounded-xl flex items-center justify-center mb-4 mx-auto">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold mb-2">高效处理</h3>
                <p className="text-gray-600 text-sm">基于深度学习的快速水印嵌入和提取</p>
              </div>
              
              <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:shadow-xl transition-all duration-300">
                <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center mb-4 mx-auto">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold mb-2">智能分析</h3>
                <p className="text-gray-600 text-sm">提供详细的数据可视化和分析报告</p>
              </div>
            </div>

            {/* 数据库选择 */}
            <div className="max-w-2xl mx-auto">
              <h2 className="text-2xl font-bold mb-8 text-gray-800">选择您的向量数据库</h2>
              <div className="grid md:grid-cols-2 gap-6">
                <Link
                  to="/pgvector"
                  className="group relative overflow-hidden bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:scale-105"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-teal-500/10 to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  <div className="relative">
                    <div className="w-16 h-16 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-2xl flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-300">
                      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 1.79 4 4 4h8c2.21 0 4-1.79 4-4V7c0-2.21-1.79-4-4-4H8c-2.21 0-4 1.79-4 4z" />
                      </svg>
                    </div>
                    <h3 className="text-xl font-bold mb-3 text-gray-800">PGVector</h3>
                    <p className="text-gray-600 mb-4">PostgreSQL向量扩展，成熟稳定的关系型数据库解决方案</p>
                    <div className="flex items-center justify-center text-teal-600 font-medium">
                      开始使用
                      <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  </div>
                </Link>

                <Link
                  to="/milvus"
                  className="group relative overflow-hidden bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:scale-105"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-pink-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  <div className="relative">
                    <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-300">
                      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                    </div>
                    <h3 className="text-xl font-bold mb-3 text-gray-800">Milvus</h3>
                    <p className="text-gray-600 mb-4">专为向量数据设计的高性能数据库，支持大规模向量检索</p>
                    <div className="flex items-center justify-center text-purple-600 font-medium">
                      开始使用
                      <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  </div>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen">
        <Navigation />
        
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/pgvector" element={<PgvectorPage />} />
            <Route path="/milvus" element={<MilvusPage />} />
          </Routes>
        </main>

        {/* 页脚 */}
        <footer className="bg-gray-900 text-white py-12">
          <div className="container mx-auto px-6">
            <div className="grid md:grid-cols-4 gap-8">
              <div className="md:col-span-2">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-bold">数据库水印系统</h3>
                </div>
                <p className="text-gray-400 mb-4">
                  基于深度学习的向量数据库水印技术，为您的数据提供安全可靠的版权保护。
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold mb-4">产品</h4>
                <ul className="space-y-2 text-gray-400">
                  <li><Link to="/pgvector" className="hover:text-white transition-colors">PGVector</Link></li>
                  <li><Link to="/milvus" className="hover:text-white transition-colors">Milvus</Link></li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold mb-4">支持</h4>
                <ul className="space-y-2 text-gray-400">
                  <li><a href="#" className="hover:text-white transition-colors">文档</a></li>
                  <li><a href="#" className="hover:text-white transition-colors">帮助中心</a></li>
                </ul>
              </div>
            </div>
            
            <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
              <p>&copy; 2025 数据库水印系统. All rights reserved.</p>
            </div>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}