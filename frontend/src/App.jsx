import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import PgvectorPage from './pages/PgvectorPage';
// 移除 import MilvusPage from './pages/MilvusPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex flex-col min-h-screen">
        {/* 顶部导航 */}
        <nav className="bg-white shadow">
          <div className="container mx-auto relative px-6 py-4">
            {/* Home 按钮放在左侧 */}
            <Link
              to="/"
              className="absolute left-6 top-1/2 transform -translate-y-1/2 text-gray-600 hover:text-blue-600 transition"
            >
              Home
            </Link>
            {/* 标题居中 */}
            <div className="text-2xl font-bold text-blue-600 text-center">
              VectorWM Demo
            </div>
          </div>
        </nav>

        {/* 主内容区 */}
        <main className="flex-grow container mx-auto px-6 py-12">
          <Routes>
            {/* 首页：居中展示单个按钮 */}
            <Route
              path="/"
              element={
                <div className="flex flex-col items-center justify-center h-full">
                  <h1 className="text-4xl font-extrabold mb-8 text-gray-800">
                    请选择向量数据库
                  </h1>
                  <div>
                    <Link
                      to="/pgvector"
                      className="px-8 py-3 bg-green-500 hover:bg-green-600 text-white rounded-lg shadow"
                    >
                      PGVector
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
        <footer className="bg-white text-center py-4 shadow-inner">
          <p className="text-sm text-gray-500">© 2025 VectorWM</p>
        </footer>
      </div>
    </BrowserRouter>
  );
}