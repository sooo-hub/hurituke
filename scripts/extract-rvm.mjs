// RVM (Robust Video Matting) を使ったオフライン抽出の精度検証用スクリプト。
// 録画済み動画から人物を抜き出し、背景画像と合成した動画を書き出す。
//
// 使い方:
//   node scripts/extract-rvm.mjs --in input.webm --bg public/bg.svg --out out.mp4

import { spawn } from 'node:child_process'
import { once } from 'node:events'
import path from 'node:path'
import ffmpegPath from 'ffmpeg-static'
import ffprobePath from 'ffprobe-static'
import sharp from 'sharp'
import * as ort from 'onnxruntime-node'

const MODEL_PATH = path.resolve('models/rvm_mobilenetv3_fp32.onnx')

function parseArgs(argv) {
  const args = { downsample: null }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--in') args.in = argv[++i]
    else if (a === '--bg') args.bg = argv[++i]
    else if (a === '--out') args.out = argv[++i]
    else if (a === '--downsample') args.downsample = parseFloat(argv[++i])
  }
  if (!args.in || !args.bg || !args.out) {
    throw new Error('使い方: node scripts/extract-rvm.mjs --in <input> --bg <background> --out <output.mp4>')
  }
  return args
}

function probeVideo(inputPath) {
  return new Promise((resolve, reject) => {
    const proc = spawn(ffprobePath.path, [
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate',
      '-of', 'json',
      inputPath,
    ])
    let out = ''
    proc.stdout.on('data', (d) => (out += d))
    proc.on('close', (code) => {
      if (code !== 0) return reject(new Error(`ffprobe failed (code ${code})`))
      const info = JSON.parse(out).streams[0]
      // MediaRecorderのwebmは可変フレームレートのことが多く、r_frame_rateが
      // 異常値 (例: 1000/1) になりがちなので avg_frame_rate を優先し、
      // それでも不自然な値なら30fpsにフォールバックする
      const parseRate = (s) => {
        const [num, den] = s.split('/').map(Number)
        return den ? num / den : NaN
      }
      let fps = parseRate(info.avg_frame_rate)
      if (!fps || fps < 1 || fps > 120) fps = parseRate(info.r_frame_rate)
      if (!fps || fps < 1 || fps > 120) fps = 30
      resolve({ width: info.width, height: info.height, fps })
    })
  })
}

async function loadBackground(bgPath, width, height) {
  const buf = await sharp(bgPath).resize(width, height).ensureAlpha().raw().toBuffer()
  // RGBA -> RGB float [0,1]
  const rgb = new Float32Array(width * height * 3)
  for (let i = 0, j = 0; i < width * height; i++, j += 4) {
    rgb[i] = buf[j] / 255
    rgb[width * height + i] = buf[j + 1] / 255
    rgb[width * height * 2 + i] = buf[j + 2] / 255
  }
  return rgb
}

function zeroTensor() {
  return new ort.Tensor('float32', new Float32Array([0]), [1, 1, 1, 1])
}

async function main() {
  const args = parseArgs(process.argv.slice(2))

  console.log('[extract-rvm] 動画情報を取得中...')
  const { width, height, fps } = await probeVideo(args.in)
  console.log(`[extract-rvm] ${width}x${height} @ ${fps.toFixed(2)}fps`)

  const downsampleRatio = args.downsample ?? Math.min(1, 512 / Math.min(width, height))
  console.log(`[extract-rvm] downsample_ratio = ${downsampleRatio.toFixed(3)}`)

  console.log('[extract-rvm] 背景画像を読み込み中...')
  const bgRgb = await loadBackground(args.bg, width, height)

  console.log('[extract-rvm] ONNXモデルを読み込み中...')
  const session = await ort.InferenceSession.create(MODEL_PATH)

  const frameSize = width * height * 3
  const readProc = spawn(ffmpegPath, [
    '-i', args.in,
    // ブラウザ録画 (可変フレームレート) を固定fpsにリサンプルしてからデコードする。
    // これをしないと入力のタイムスタンプが乱れている場合に想定外の枚数のフレームが
    // デコードされ、処理時間が跳ね上がることがある。
    '-r', String(fps),
    '-f', 'rawvideo',
    '-pix_fmt', 'rgb24',
    '-',
  ])
  const writeProc = spawn(ffmpegPath, [
    '-y',
    '-f', 'rawvideo',
    '-pix_fmt', 'rgb24',
    '-s', `${width}x${height}`,
    '-r', String(fps),
    '-i', '-',
    '-an',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    args.out,
  ])
  writeProc.stderr.on('data', () => {})
  readProc.stderr.on('data', () => {})

  let r1i = zeroTensor()
  let r2i = zeroTensor()
  let r3i = zeroTensor()
  let r4i = zeroTensor()
  const dsTensor = new ort.Tensor('float32', new Float32Array([downsampleRatio]), [1])

  let pending = Buffer.alloc(0)
  let frameIndex = 0
  let writeQueue = Promise.resolve()

  readProc.stdout.on('data', (chunk) => {
    pending = pending.length ? Buffer.concat([pending, chunk]) : chunk
    while (pending.length >= frameSize) {
      const frameBuf = pending.subarray(0, frameSize)
      pending = pending.subarray(frameSize)
      const frame = Buffer.from(frameBuf) // copy out before buffer is reused

      readProc.stdout.pause()
      writeQueue = writeQueue
        .then(() => processFrame(frame))
        .then(() => {
          frameIndex++
          if (frameIndex % 30 === 0) console.log(`[extract-rvm] ${frameIndex} フレーム処理済み`)
          readProc.stdout.resume()
        })
    }
  })

  async function processFrame(rgbBuf) {
    // HWC(uint8) -> CHW(float32, 0-1)
    const src = new Float32Array(3 * width * height)
    const plane = width * height
    for (let i = 0, j = 0; i < plane; i++, j += 3) {
      src[i] = rgbBuf[j] / 255
      src[plane + i] = rgbBuf[j + 1] / 255
      src[plane * 2 + i] = rgbBuf[j + 2] / 255
    }
    const srcTensor = new ort.Tensor('float32', src, [1, 3, height, width])

    const results = await session.run({
      src: srcTensor,
      r1i, r2i, r3i, r4i,
      downsample_ratio: dsTensor,
    })

    r1i = results.r1o
    r2i = results.r2o
    r3i = results.r3o
    r4i = results.r4o

    const fgr = results.fgr.data
    const pha = results.pha.data

    // 合成: out = fgr * pha + bg * (1 - pha)
    const outBuf = Buffer.alloc(frameSize)
    for (let i = 0, j = 0; i < plane; i++, j += 3) {
      const a = pha[i]
      outBuf[j] = Math.round((fgr[i] * a + bgRgb[i] * (1 - a)) * 255)
      outBuf[j + 1] = Math.round((fgr[plane + i] * a + bgRgb[plane + i] * (1 - a)) * 255)
      outBuf[j + 2] = Math.round((fgr[plane * 2 + i] * a + bgRgb[plane * 2 + i] * (1 - a)) * 255)
    }

    if (!writeProc.stdin.write(outBuf)) {
      await once(writeProc.stdin, 'drain')
    }
  }

  await once(readProc, 'close')
  await writeQueue
  writeProc.stdin.end()
  await once(writeProc, 'close')

  console.log(`[extract-rvm] 完了: ${frameIndex} フレーム -> ${args.out}`)
}

main().catch((err) => {
  console.error('[extract-rvm] エラー:', err)
  process.exit(1)
})
