/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'formal': {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
        },
        'polite': {
          50: '#f0fdf4',
          500: '#22c55e',
          600: '#16a34a',
        },
        'casual': {
          50: '#fff7ed',
          500: '#f97316',
          600: '#ea580c',
        }
      }
    },
  },
  plugins: [],
}
