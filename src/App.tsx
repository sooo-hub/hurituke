import { useEffect, useRef, useState } from 'react'
import { createSegmenter } from './segmenter'
import type { ImageSegmenter } from '@mediapipe/tasks-vision'

const VIDEO_W = 640
const VIDEO_H = 480
// 新フレームの反映度合い (小さいほど滑らかだが追従が遅れる)
const TEMPORAL_SMOOTHING = 0.6
// マスク値の底上げカーブ (1未満で中間〜低信頼度を持ち上げる。指先など細い部位の欠け対策)
const MASK_GAMMA = 0.5

type Status =
  | 'init'
  | 'requesting-camera'
  | 'loading-model'
  | 'running'
  | 'error'

const STATUS_LABEL: Record<Status, string> = {
  init: '初期化中...',
  'requesting-camera': 'カメラ権限を要求中...',
  'loading-model': 'AIモデルを読み込み中（初回は時間がかかります）...',
  running: '動作中',
  error: 'エラー',
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // ピクセル操作用のオフスクリーンバッファ
  const bufferCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const segmenterRef = useRef<ImageSegmenter | null>(null)
  const bgImageRef = useRef<HTMLImageElement | null>(null)
  const rafRef = useRef<number>(0)
  // フレーム間のマスク値スムージング用 (ちらつき対策)
  const smoothedMaskRef = useRef<Float32Array | null>(null)

  const [status, setStatus] = useState<Status>('init')
  const [fps, setFps] = useState(0)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    let cancelled = false

    async function init() {
      try {
        // ── Step 1: カメラ取得 ───────────────────────────
        setStatus('requesting-camera')
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: VIDEO_W, height: VIDEO_H, facingMode: 'user' },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }

        const video = videoRef.current!
        video.srcObject = stream
        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => resolve()
        })
        await video.play()

        // ── Step 2: 背景画像読み込み ──────────────────────
        const bg = new Image()
        bg.src = '/bg.svg'
        await new Promise<void>((resolve, reject) => {
          bg.onload = () => resolve()
          bg.onerror = () => reject(new Error('背景画像の読み込みに失敗しました'))
        })
        bgImageRef.current = bg

        // ── Step 3: MediaPipe セグメンター初期化 ──────────
        setStatus('loading-model')
        const segmenter = await createSegmenter()
        if (cancelled) {
          segmenter.close()
          return
        }
        segmenterRef.current = segmenter

        // ピクセル操作用オフスクリーンキャンバスを生成
        const buffer = document.createElement('canvas')
        buffer.width = VIDEO_W
        buffer.height = VIDEO_H
        bufferCanvasRef.current = buffer

        setStatus('running')

        // ── Step 4: レンダリングループ ────────────────────
        let frameCount = 0
        let fpsOrigin = 0

        function loop(timestamp: number) {
          if (cancelled) return

          const video = videoRef.current
          const canvas = canvasRef.current
          const buffer = bufferCanvasRef.current
          const segmenter = segmenterRef.current
          const bg = bgImageRef.current

          if (
            !video ||
            !canvas ||
            !buffer ||
            !segmenter ||
            !bg ||
            video.readyState < 2
          ) {
            rafRef.current = requestAnimationFrame(loop)
            return
          }

          const ctx = canvas.getContext('2d')!
          const bctx = buffer.getContext('2d')!

          // セグメンテーション実行
          const result = segmenter.segmentForVideo(video, timestamp)
          // このモデル (selfie_segmenter float16/1) は confidenceMasks を1枚だけ返し、
          // index 0 が人物の信頼度 (0.0〜1.0)
          const masks = result.confidenceMasks
          const personMask = masks && masks[masks.length - 1]

          if (personMask) {
            // 映像フレームをバッファに描画
            bctx.drawImage(video, 0, 0, VIDEO_W, VIDEO_H)
            const imageData = bctx.getImageData(0, 0, VIDEO_W, VIDEO_H)
            const pixels = imageData.data
            // Float32Array: 人物=1.0, 背景=0.0
            const maskData = personMask.getAsFloat32Array()
            const pixelCount = maskData.length

            // 前フレームの値とブレンドしてちらつきを抑える
            let smoothed = smoothedMaskRef.current
            if (!smoothed || smoothed.length !== pixelCount) {
              smoothed = new Float32Array(maskData)
              smoothedMaskRef.current = smoothed
            }

            // マスク値をアルファチャンネルに適用 (滑らかな輪郭)
            for (let i = 0; i < pixelCount; i++) {
              const value = smoothed[i] * (1 - TEMPORAL_SMOOTHING) + maskData[i] * TEMPORAL_SMOOTHING
              smoothed[i] = value
              // 中間信頼度を持ち上げ、腕など輪郭が薄くなりがちな部分の欠けを軽減
              pixels[i * 4 + 3] = Math.round(Math.pow(value, MASK_GAMMA) * 255)
            }
            bctx.putImageData(imageData, 0, 0)

            // 合成: 背景 → 人物
            ctx.drawImage(bg, 0, 0, VIDEO_W, VIDEO_H)
            ctx.drawImage(buffer, 0, 0)

            personMask.close()
          }

          result.close()

          // FPS計測
          frameCount++
          if (fpsOrigin === 0) fpsOrigin = timestamp
          const elapsed = timestamp - fpsOrigin
          if (elapsed >= 1000) {
            setFps(Math.round((frameCount * 1000) / elapsed))
            frameCount = 0
            fpsOrigin = timestamp
          }

          rafRef.current = requestAnimationFrame(loop)
        }

        rafRef.current = requestAnimationFrame(loop)
      } catch (err) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : String(err)
          setError(msg)
          setStatus('error')
        }
      }
    }

    init()

    return () => {
      cancelled = true
      cancelAnimationFrame(rafRef.current)
      segmenterRef.current?.close()
      // カメラストリームを停止
      const video = videoRef.current
      if (video?.srcObject instanceof MediaStream) {
        video.srcObject.getTracks().forEach((t) => t.stop())
      }
    }
  }, [])

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Dance Composite — Prototype</h1>

      <div style={styles.statusBar}>
        <span
          style={{
            ...styles.statusDot,
            backgroundColor: status === 'running' ? '#4caf50' : status === 'error' ? '#f44336' : '#ff9800',
          }}
        />
        <span>{STATUS_LABEL[status]}</span>
        {status === 'running' && fps > 0 && (
          <span style={styles.fps}>{fps} FPS</span>
        )}
      </div>

      {error && <p style={styles.error}>エラー: {error}</p>}

      {/* 合成結果キャンバス */}
      <canvas
        ref={canvasRef}
        width={VIDEO_W}
        height={VIDEO_H}
        style={styles.canvas}
      />

      {/* 非表示のカメラ映像 */}
      <video
        ref={videoRef}
        width={VIDEO_W}
        height={VIDEO_H}
        muted
        playsInline
        style={{ display: 'none' }}
      />

      <p style={styles.hint}>
        ※ カメラ権限を許可してください。モデル初回読み込みに10〜20秒かかります。
      </p>
    </div>
  )
}

const styles = {
  container: {
    minHeight: '100vh',
    background: '#111',
    color: '#eee',
    fontFamily: 'system-ui, sans-serif',
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    gap: '16px',
    padding: '24px 16px',
  },
  title: {
    fontSize: '1.4rem',
    fontWeight: 600,
    color: '#ddd',
    letterSpacing: '0.05em',
  },
  statusBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '0.9rem',
    color: '#aaa',
  },
  statusDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    display: 'inline-block',
  },
  fps: {
    marginLeft: '8px',
    padding: '2px 8px',
    background: '#222',
    borderRadius: '4px',
    fontFamily: 'monospace',
    fontSize: '0.85rem',
    color: '#4caf50',
  },
  error: {
    color: '#f44336',
    background: '#2a0a0a',
    padding: '8px 16px',
    borderRadius: '4px',
    maxWidth: '640px',
    wordBreak: 'break-word' as const,
  },
  canvas: {
    border: '1px solid #333',
    borderRadius: '4px',
    maxWidth: '100%',
    display: 'block',
  },
  hint: {
    fontSize: '0.78rem',
    color: '#666',
    textAlign: 'center' as const,
    maxWidth: '640px',
  },
}
