// web_ui/src/pages/MilvusPage.jsx
import React, { useState } from 'react';
import { getLowDegreeVectors } from '../api';

export default function MilvusPage() {
  const [n, setN] = useState(10);
  const [ids, setIds] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchMilvusVectors = async () => {
    setLoading(true);
    try {
      // TODO: 替换为真正的 Milvus 接口
      const list = await getLowDegreeVectors(n);
      setIds(list);
    } catch (err) {
      console.error(err);
      alert('获取 Milvus 向量失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8">
      <h2 className="text-3xl font-bold mb-6">Milvus 向量操作</h2>

      <section className="mb-8">
        <label className="block mb-2 font-medium">要取的向量数量 (N)</label>
        <div className="flex items-center">
          <input
            type="number"
            min={1}
            value={n}
            onChange={e => setN(parseInt(e.target.value) || 1)}
            className="border rounded px-3 py-2 w-24 mr-4"
          />
          <button
            onClick={fetchMilvusVectors}
            disabled={loading}
            className="bg-purple-600 hover:bg-purple-700 text-white px-5 py-2 rounded disabled:opacity-50"
          >
            {loading ? '加载中…' : '获取向量'}
          </button>
        </div>
      </section>

      {ids.length > 0 && (
        <section>
          <h3 className="text-2xl font-semibold mb-4">返回的向量 ID 列表</h3>
          <ul className="grid grid-cols-2 gap-2">
            {ids.map(id => (
              <li
                key={id}
                className="bg-purple-50 border border-purple-200 rounded px-4 py-2"
              >
                ID: {id}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
