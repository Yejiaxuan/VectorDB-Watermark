import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import PgvectorPage from './pages/PgvectorPage';
// 移除 import MilvusPage from './pages/MilvusPage';

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
            {/* 首页：居中展示单个按钮 */}
            <Route
              path="/"
              element={
                <div className="flex flex-col items-center justify-center h-full min-h-96">
                  <h1 className="text-4xl font-bold mb-8 text-gray-800">
                    请选择向量数据库
                  </h1>
                  <div>
                    <Link
                      to="/pgvector"
                      className="inline-flex items-center px-8 py-4 
                               bg-gradient-to-r from-teal-500 to-cyan-500 
                               hover:from-teal-600 hover:to-cyan-600 
                               text-white font-semibold rounded-xl shadow-lg 
                               hover:shadow-xl transform hover:scale-105 
                               transition-all duration-100 ease-in-out"
                    >
                      PGVector 数据库
                    </Link>
                  </div>
                </div>
              }
            />
            <Route path="/pgvector" element={<PgvectorPage />} />
            {/* 移除 <Route path="/milvus" element={<MilvusPage />} /> */}
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