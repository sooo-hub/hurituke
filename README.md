# hurituke (dance-composite)

Webカメラの映像から MediaPipe の Selfie Segmentation で人物領域を抽出し、
背景画像と合成してブラウザ上にリアルタイム表示するプロトタイプです。

## セットアップ

```bash
npm install
npm run dev
```

ブラウザでカメラへのアクセスを許可してください。AIモデルの初回読み込みに10〜20秒ほどかかります。

## スクリプト

- `npm run dev` — 開発サーバーを起動
- `npm run build` — 型チェック後に本番ビルド
- `npm run preview` — ビルド結果をローカルでプレビュー

## 技術構成

- React + TypeScript + Vite
- [@mediapipe/tasks-vision](https://www.npmjs.com/package/@mediapipe/tasks-vision) の `ImageSegmenter`（selfie segmenter モデル）を使用し、`getUserMedia` の映像から人物マスクをフレームごとに算出
- 抽出した人物マスクを `public/bg.svg` の背景画像に合成し、`<canvas>` に描画

## オフライン抽出 (精度検証用)

リアルタイムのブラウザ処理は selfie segmenter の精度上限（指先などの細い部位が欠けやすい）があるため、
録画してから [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) で
オフライン抽出する精度検証用スクリプトを用意しています。最終的な配布アーキテクチャは未確定で、
これは精度比較のための実装です。

1. アプリの「● 録画開始」でカメラ映像を録画し、「録画をダウンロード」で `.webm` を保存
2. RVMモデル（`rvm_mobilenetv3_fp32.onnx`, 約14MB）を取得し `models/` に配置
   ```bash
   mkdir -p models
   curl -L -o models/rvm_mobilenetv3_fp32.onnx https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
   ```
3. 抽出を実行
   ```bash
   npm run extract-rvm -- --in <録画ファイル> --bg public/bg.svg --out out.mp4
   ```

CPU実行・ダウンサンプルなしの場合、目安として15秒の動画で数分かかります（`--downsample <0-1>` で速度と引き換えに精度を下げられます）。
