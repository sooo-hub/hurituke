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
