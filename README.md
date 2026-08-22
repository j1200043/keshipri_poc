# 『消しプリ』AIモデル検証(PoC)

「撮影したプリントから手書き文字を自動消去する」機能を再現・再検証するためのリポジトリです。役割の異なる**2つの独立したパイプライン**で構成されています。混同しないよう、それぞれ別のドキュメント・Label Studioプロジェクト・スクリプト命名にしています。

## パイプライン一覧

| パイプライン | 目的 | ドキュメント |
|---|---|---|
| **手書き領域検出**(YOLO) | 手書きが「どこにあるか」を矩形で検出する | [README_YOLO_DETECTION.md](README_YOLO_DETECTION.md) |
| **手書きピクセル抽出**(セグメンテーション) | 検出領域内の「どのピクセルが手書きか」を判定する | [README_SEGMENTATION.md](README_SEGMENTATION.md) |

両者は実際のプリント消去パイプラインの中で連携する想定です(検出 → その領域内でピクセル単位の消去)。それぞれ入力データ・Label Studioプロジェクト・学習/変換スクリプトを分けているので、作業する際は必ず該当ドキュメントの指示に従ってください。

## 環境構築(共通)

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

以降の詳細な手順(データ準備 → アノテーション → 学習 → CoreML変換)は各パイプラインのドキュメントを参照してください。
