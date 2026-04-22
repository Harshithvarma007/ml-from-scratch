import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Terminal Color System — Claude Code inspired (ported from Octa)
        'dark-bg': '#0a0a0a',
        'dark-sidebar': '#111111',
        'dark-surface': '#111111',
        'dark-surface-elevated': '#161616',
        'dark-surface-hover': '#1c1c1c',
        'dark-input': '#0a0a0a',

        'dark-border': '#1e1e1e',
        'dark-border-hover': '#2e2e2e',
        'dark-border-focus': '#a78bfa',

        'dark-text-primary': '#e8e8e8',
        'dark-text-secondary': '#888888',
        'dark-text-muted': '#555555',
        'dark-text-disabled': '#333333',

        'dark-accent': '#a78bfa',      // Claude purple
        'dark-accent-hover': '#9670f0',

        'dark-sql': '#60a5fa',
        'dark-python': '#4ade80',
        'dark-magic': '#c084fc',
        'dark-success': '#4ade80',
        'dark-warning': '#fbbf24',
        'dark-error': '#f87171',
        'dark-info': '#60a5fa',

        'dark-scrollbar': '#1e1e1e',
        'dark-scrollbar-hover': '#2e2e2e',

        // Terminal-specific
        'term-green': '#4ade80',
        'term-purple': '#a78bfa',
        'term-cyan': '#67e8f9',
        'term-pink': '#f472b6',
        'term-amber': '#fbbf24',
        'term-teal': '#5eead4',
        'term-rose': '#fb7185',
        'term-orange': '#fb923c',
        'term-indigo': '#818cf8',
        'term-fuchsia': '#e879f9',
        'term-emerald': '#34d399',
        'term-slate': '#94a3b8',
        'term-dim': '#444444',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Menlo', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'fade-up': 'fadeUp 0.4s ease-out forwards',
        'slide-in': 'slideIn 0.3s ease-out forwards',
        'scale-in': 'scaleIn 0.2s ease-out forwards',
        'shimmer': 'shimmer 1.5s ease-in-out infinite',
        'float': 'float 3s ease-in-out infinite',
        'blink': 'blink 1s step-end infinite',
        'slide-left': 'slideLeft 0.2s ease-out forwards',
      },
      keyframes: {
        glow: {
          '0%': { opacity: '1' },
          '100%': { opacity: '0.5' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideIn: {
          '0%': { opacity: '0', transform: 'translateX(-16px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-6px)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        slideLeft: {
          '0%': { opacity: '0', transform: 'translateX(-12px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
      },
      transitionDuration: {
        'fast': '150ms',
        'normal': '250ms',
        'slow': '400ms',
      },
      boxShadow: {
        'dark-sm': '0 1px 3px 0 rgba(0, 0, 0, 0.5)',
        'dark-md': '0 4px 12px 0 rgba(0, 0, 0, 0.4)',
        'dark-lg': '0 10px 40px 0 rgba(0, 0, 0, 0.6)',
        'dark-glow': '0 0 0 1px rgba(167, 139, 250, 0.3), 0 2px 8px 0 rgba(167, 139, 250, 0.15)',
        'dark-glow-strong': '0 0 0 2px rgba(167, 139, 250, 0.5), 0 4px 16px 0 rgba(167, 139, 250, 0.3)',
      },
    },
  },
  plugins: [],
}
export default config
