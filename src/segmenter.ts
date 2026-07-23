import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision'

// WASMはjsDelivrのCDNから取得
const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'

// MediaPipe 公式ホスト上のselfieセグメンターモデル
// カテゴリ: 0=背景, 1=人物
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite'

export async function createSegmenter(): Promise<ImageSegmenter> {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL)
  const segmenter = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    // confidenceMasks を使うと輪郭が滑らかになる (float32: 0.0〜1.0)
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  })
  return segmenter
}
