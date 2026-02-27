/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // DeepBlue Command Deck 配色方案
        'deep': {
          '950': '#0a0e1a',
          '900': '#0f1420',
          '800': '#1a1f2e'
        },
        'neon-blue': '#06b6d4',
        'neon-green': '#10b981',
        'neon-purple': '#a855f7'
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'monospace']
      },
      animation: {
        'pulse': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce': 'bounce 1s infinite'
      }
    }
  },
  plugins: []
}
