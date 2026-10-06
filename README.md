# 生物ミキサー

2種類の生物を選び、スライダーで中間の姿を眺めて遊ぶ小さな静的サイト（技術デモ兼おもちゃ）。

- **C：事前生成** … 中間画像をPCで事前に生成し、ブラウザでは切り替えるだけ（隣接フレームはクロスフェード）
- **A：ブラウザでデコード** … 事前計算した latent をブラウザで補間し、TAESD / TAESDXL（ONNX Runtime Web、WebGPU優先）でデコード

## スタイル
| スタイル | 生成モデル | ブラウザ用デコーダ |
|---|---|---|
| ファンタジーアニメ | [Animagine XL 4.0](https://huggingface.co/cagliostrolab/animagine-xl-4.0)（CreativeML Open RAIL++-M） | TAESDXL |
| ダークファンタジー／絵本・水彩風 | [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)（Stability AI） | TAESD |

デコーダ：[TAESD / TAESDXL](https://github.com/madebyollin/taesd)（MIT）をONNXに変換したもの。

## 開発用
URLに `?dev` を付けると、補間方式（Linear / Slerp）とデコード時間が表示されます。
