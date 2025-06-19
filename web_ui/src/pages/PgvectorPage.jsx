import React, { useState } from 'react';

export default function PgvectorPage() {
  // 连接数据库配置状态
  const [ip, setIp] = useState('localhost');
  const [port, setPort] = useState(5432);
  const [dbname, setDbname] = useState('test');
  const [user, setUser] = useState('postgres');
  const [password, setPassword] = useState('');
  const [connected, setConnected] = useState(false);
  const [statusMsg, setStatusMsg] = useState('未连接');

  // 水印操作状态
  const [message, setMessage] = useState('');
  const [embedResult, setEmbedResult] = useState('');
  const [extractResult, setExtractResult] = useState('');

  // 模拟连接数据库
  const handleConnect = () => {
    // TODO: 在这里调用后端连接接口
    // stub 直接标记为已连接
    setConnected(true);
    setStatusMsg(`已连接 ${ip}:${port}/${dbname} as ${user}`);
  };

  // 模拟嵌入水印
  const handleEmbed = () => {
    if (!connected) return;
    // TODO: 调用 /watermark/embed API
    setEmbedResult(`已对消息 "${message}" 进行嵌入`);
  };

  // 模拟提取水印
  const handleExtract = () => {
    if (!connected) return;
    // TODO: 调用 /watermark/extract API
    setExtractResult(`提取到消息："${message}"`);
  };

  return (
    <div className="p-8 grid grid-cols-2 gap-8">
      {/* 左侧：数据库连接配置 */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-2xl font-semibold mb-4">PGVector 连接配置</h2>
        <div className="space-y-3">
          <div>
            <label className="block font-medium">Host / IP</label>
            <input
              type="text"
              value={ip}
              onChange={e => setIp(e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
          <div>
            <label className="block font-medium">Port</label>
            <input
              type="number"
              value={port}
              onChange={e => setPort(parseInt(e.target.value) || 0)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
          <div>
            <label className="block font-medium">Database</label>
            <input
              type="text"
              value={dbname}
              onChange={e => setDbname(e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
          <div>
            <label className="block font-medium">User</label>
            <input
              type="text"
              value={user}
              onChange={e => setUser(e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
          <div>
            <label className="block font-medium">Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
          <button
            onClick={handleConnect}
            className="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded disabled:opacity-50"
          >
            {connected ? '已连接' : '连接数据库'}
          </button>
        </div>
        <p className="mt-4 text-sm text-gray-600">状态：{statusMsg}</p>
      </div>

      {/* 右侧：水印操作 */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-2xl font-semibold mb-4">水印嵌入与提取</h2>
        <div className="space-y-3">
          <div>
            <label className="block font-medium">消息内容</label>
            <textarea
              rows={3}
              value={message}
              onChange={e => setMessage(e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
          <div className="flex space-x-4">
            <button
              onClick={handleEmbed}
              disabled={!connected || !message}
              className="flex-1 bg-green-500 hover:bg-green-600 text-white py-2 rounded disabled:opacity-50"
            >
              嵌入水印
            </button>
            <button
              onClick={handleExtract}
              disabled={!connected}
              className="flex-1 bg-purple-500 hover:bg-purple-600 text-white py-2 rounded disabled:opacity-50"
            >
              提取水印
            </button>
          </div>
        </div>
        {embedResult && (
          <p className="mt-4 text-green-700">{embedResult}</p>
        )}
        {extractResult && (
          <p className="mt-2 text-purple-700">{extractResult}</p>
        )}
      </div>
    </div>
  );
}
