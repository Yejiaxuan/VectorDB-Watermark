/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",           // 如果你在 index.html 里也用 Tailwind
    "./src/**/*.{js,jsx}",    // 让 Tailwind 扫描 src 下所有 .js/.jsx 文件
  ],
  theme: {
    extend: {
      animation: {
        "fade-in": "fadeIn 200ms ease-in-out",
        "slide-in-left": "slideInLeft 200ms ease-in-out",
        "slide-in-right": "slideInRight 200ms ease-in-out",
        "fill-line": "fillLine 300ms ease-in-out forwards",
        "pulse-slow": "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "bounce-subtle": "bounceSubtle 0.5s ease-in-out",
        "scale-in": "scaleIn 150ms ease-in-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideInLeft: {
          "0%": { opacity: "0", transform: "translateX(-16px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        slideInRight: {
          "0%": { opacity: "0", transform: "translateX(16px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        fillLine: {
          "0%": { width: "0%" },
          "100%": { width: "100%" },
        },
        bounceSubtle: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-2px)" },
        },
        scaleIn: {
          "0%": { transform: "scale(0.95)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
      },
      backdropBlur: {
        'xs': '2px',
      },
    },
  },
  plugins: [],
}

