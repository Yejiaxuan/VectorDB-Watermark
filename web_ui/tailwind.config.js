/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",           // 如果你在 index.html 里也用 Tailwind
    "./src/**/*.{js,jsx}",    // 让 Tailwind 扫描 src 下所有 .js/.jsx 文件
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}

