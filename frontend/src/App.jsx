import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import PgvectorPage from './pages/PgvectorPage';
import MilvusPage from './pages/MilvusPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex flex-col min-h-screen bg-gray-50">
        {/* 顶部导航 */}
        <nav className="bg-white shadow-sm border-b border-gray-100">
          <div className="container mx-auto relative px-6 py-4">
            {/* Home 按钮放在左侧 */}
            <Link
              to="/"
              className="absolute left-6 top-1/2 transform -translate-y-1/2 
                         text-gray-600 hover:text-teal-600 transition-colors duration-200 
                         font-medium"
            >
              主页
            </Link>
            {/* 标题居中 */}
            <div className="text-2xl font-bold bg-gradient-to-r from-teal-600 to-cyan-600 
                           bg-clip-text text-transparent text-center">
              数据库水印系统
            </div>
          </div>
        </nav>

        {/* 主内容区 */}
        <main className="flex-grow container mx-auto px-6 py-8">
          <Routes>
            {/* 首页：居中展示两个按钮 */}
            <Route
              path="/"
              element={
                <div className="flex flex-col items-center justify-center h-full min-h-96">
                  <h1 className="text-4xl font-bold mb-8 text-gray-800">
                    请选择向量数据库
                  </h1>
                  <div className="flex gap-6">
                    <Link
                      to="/pgvector"
                      className="inline-flex items-center px-8 py-4 
                               bg-gradient-to-r from-teal-500 to-cyan-500 
                               hover:from-teal-600 hover:to-cyan-600 
                               text-white font-semibold rounded-xl shadow-lg 
                               hover:shadow-xl transform hover:scale-105 
                               transition-all duration-100 ease-in-out"
                    >
                      <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 1.79 4 4 4h8c2.21 0 4-1.79 4-4V7c0-2.21-1.79-4-4-4H8c-2.21 0-4 1.79-4 4z" />
                      </svg>
                      PGVector 数据库
                    </Link>
                    <Link
                      to="/milvus"
                      className="inline-flex items-center px-8 py-4 
                               bg-gradient-to-r from-purple-500 to-pink-500 
                               hover:from-purple-600 hover:to-pink-600 
                               text-white font-semibold rounded-xl shadow-lg 
                               hover:shadow-xl transform hover:scale-105 
                               transition-all duration-100 ease-in-out"
                    >
                      <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                      Milvus 数据库
                    </Link>
                  </div>
                </div>
              }
            />
            <Route path="/pgvector" element={<PgvectorPage />} />
            <Route path="/milvus" element={<MilvusPage />} />
          </Routes>
        </main>

        {/* 页脚 */}
        <footer className="bg-white border-t border-gray-100 text-center py-6">
          <p className="text-sm text-gray-500">© 2025 数据库水印系统</p>
        </footer>
      </div>
    </BrowserRouter>
  );
}