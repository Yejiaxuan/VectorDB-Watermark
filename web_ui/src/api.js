// web_ui/src/api.js
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
