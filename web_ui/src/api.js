// web_ui/src/api.js
export async function connectDB(dbParams) {
  const res = await fetch("/api/connect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(dbParams),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "连接失败");
  }
  return res.json();  // { success: true, message: "连接成功" }
}
/**
 * 模拟从后端取入度最低的向量 ID 列表
 * @param {number} n
 * @returns {Promise<number[]>}
 */
export async function getLowDegreeVectors(n) {
  // 这里是模拟网络延迟
  await new Promise(r => setTimeout(r, 500));
  // 返回 [1,2,3,…,n]
  return Array.from({ length: n }, (_, i) => i + 1);
}
