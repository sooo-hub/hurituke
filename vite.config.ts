import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    // MediaPipe は独自のESM構造を持つため事前バンドルから除外
    exclude: ['@mediapipe/tasks-vision'],
  },
})
