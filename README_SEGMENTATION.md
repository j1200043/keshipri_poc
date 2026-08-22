# 手書きピクセル抽出(セグメンテーション)パイプライン

`cropped_images/`(YOLO検出パイプラインで切り出した手書き候補領域)を使って、**手書きピクセルだけをピクセル単位で判定するセグメンテーションモデル**を学習し、CoreMLに変換するまでの手順です。

⚠️ [README_YOLO_DETECTION.md](README_YOLO_DETECTION.md)(手書き領域の**検出**パイプライン)とは別物です。混同しないよう、スクリプト名・Label Studioプロジェクト・出力先をすべて分けています。

| | YOLO検出パイプライン | 本パイプライン(セグメンテーション) |
|---|---|---|
| 目的 | 手書きが「どこにあるか」を矩形で検出 | 検出領域内の「どのピクセルが手書きか」を判定 |
| Label Studioプロジェクト | `Keshi_YOLO`(id=4) | `Keshi_Segment`(id=5) |
| ラベリング方式 | RectangleLabels | BrushLabels |
| 設定ファイル | `label_studio_config_yolo.xml` | `label_studio_config_segmentation.xml` |
| 入力画像 | `data_prepare/images_resized/` | `data/images_letterboxed/` |
| モデル | YOLO11s | U-Net (MobileNetV2 encoder) |
| 学習スクリプト | `train.py` | `train_segmentation.py` |
| CoreML変換 | `export_coreml.py` | `export_coreml_segmentation.py` |

---

## 全体の流れ

```
cropped_images/ (元のクロップ画像、サイズ不揃い)
      │  prepare_segmentation_dataset.py
      ▼
data/images_letterboxed/ (512x320に統一, 白パディング) + manifest.json
      │  Label Studioでアノテーション(Brush)
      ▼
data/masks/ (二値マスクPNG, images_letterboxedと同名)
      │  train_segmentation.py
      ▼
models/segmentation/best.pt (PyTorchチェックポイント)
      │  export_coreml_segmentation.py
      ▼
models/segmentation/HandwritingSegmentation.mlpackage (iOS組み込み用)
```

---

## 1. 画像サイズの統一(letterbox)

`cropped_images/` は1枚1枚サイズがバラバラ(実測: 長辺89〜834px、アスペクト比1.0〜4.4)。学習のバッチ処理には固定サイズが必要なため、以下の方針で `512×320` に統一する。

- 実データの長辺p90(≒546px)・アスペクト比中央値(≒1.6)から `512×320`(32の倍数)を採用
- 縦長画像は90°回転して横長に正規化(学習側でも回転augmentationを使うため向きは非依存)
- 縮小のみ行い、小さい画像を無理に拡大しない(鉛筆線のボケ防止)
- 縮小した内容はキャンバス中央に配置し、余白は紙の色に合わせて**白パディング**
- 変換内容(スケール・パディング量・回転有無)は `data/images_letterboxed/manifest.json` に保存

```bash
python prepare_segmentation_dataset.py
```

出力: `data/images_letterboxed/*.jpg`(全て512×320)、`data/images_letterboxed/manifest.json`

---

## 2. Label Studioでのアノテーション(Brush)

### 2-1. 起動

```bash
LOCAL_FILES_SERVING_ENABLED=true LOCAL_FILES_DOCUMENT_ROOT=$(pwd) label-studio start
```

### 2-2. プロジェクト作成

- Labeling Setup → Custom template → `label_studio_config_segmentation.xml` の内容を貼り付け(ラベル名 `handwriting`, Brushツール)

### 2-3. ローカルファイルストレージの登録(初回のみ)

Label Studioは `/data/local-files/?d=...` で画像を配信する前に、対象ディレクトリを Local Files ストレージとして登録しておく必要がある(未登録だと "issue loading URL" エラーになる)。

```bash
python register_segmentation_local_storage.py <プロジェクトID>
```

UIから行う場合: プロジェクトの **Settings → Cloud Storage → Add Source Storage → Local files** で Absolute local path に `<リポジトリの絶対パス>/data/images_letterboxed` を指定。

### 2-4. タスクのインポート

```bash
python generate_labelstudio_tasks_segmentation.py
```

生成された `data/labelstudio_tasks_segmentation.json` をLabel StudioのUIで **Import** する。

### 2-5. (任意)疑似マスクで下書きを作り、作業量を減らす

暗いピクセル(Otsu二値化)+直線検出による印刷罫線の除外、というヒューリスティックで疑似マスクを自動生成し、Label StudioのPrediction(下書き)として投入できる。人が「ゼロから塗る」のではなく「間違いを直すだけ」で済むようになる。

```bash
python generate_pseudo_masks.py          # data/pseudo_masks/*.png と確認用プレビューを生成
python import_pseudo_predictions.py <プロジェクトID>   # 未アノテーションのタスクにPredictionとして投入
```

この処理は**未アノテーションのタスクのみ**が対象で、既存の手動アノテーションは上書きしない。

### 2-6. アノテーション

Brushツールで**実際に鉛筆/ペンで書かれたピクセルだけ**を塗る。印刷された罫線・枠線・グリッドは含めない。ブラシは実際の線の太さに近づけ、背景まで塗りつぶさない。

---

## 3. マスクの書き出し

アノテーション完了後、Label StudioのDBから直接マスクPNGを生成する(JSONを手動エクスポートする方法でも同じ結果になる)。

```bash
# DBから直接生成(推奨)
python export_masks_from_db.py <プロジェクトID>

# もしくは Export → JSON でファイルを落としてから
python export_masks_from_labelstudio.py <エクスポートJSONのパス>
```

出力: `data/masks/*.png`(`data/images_letterboxed/` と同名、白=手書き・黒=背景)

---

## 4. 学習

```bash
python train_segmentation.py --epochs 100
```

| 項目 | 内容 |
|---|---|
| モデル | U-Net + MobileNetV2エンコーダ(ImageNet事前学習) |
| 入力 | 512×320 |
| 損失関数 | BCE + Dice(手書きピクセルは少数派なのでクラス不均衡対策) |
| Augmentation | 上下左右反転、±15°回転(白パディング維持)、明るさ/コントラスト、ノイズ |
| データ分割 | 85% train / 15% val(`SEED=42`固定、再現可能) |
| デバイス | MPS(Apple Silicon GPU)自動検出 |
| 保存先 | `models/segmentation/best.pt`(val Diceが更新されるたびに上書き保存) |

**直近の学習結果**: val Dice = 0.884(90枚, train 77 / val 13)

---

## 5. 推論・確認

```bash
python predict_mask.py "data/images_letterboxed/*.jpg" --out-dir data/predict_output
```

各画像に対して `<name>_overlay.jpg`(検出ピクセルを赤で重ねた確認用画像)と `<name>_cleaned.jpg`(手書きピクセルを白で消した画像)を出力する。

---

## 6. CoreML変換

```bash
python export_coreml_segmentation.py
```

出力: `models/segmentation/HandwritingSegmentation.mlpackage`

- 入力: `image`(512×320のRGB画像、0〜255をそのまま。正規化はモデル内部に組み込み済み)
- 出力: `mask`(1×1×320×512、0.0〜1.0の確率マップ。Swift側で0.5前後を閾値に二値化する)
- 変換後、PyTorch版との出力差分を検証済み(平均絶対誤差 ≈ 0.00008、mlprogram形式のfp16精度による差)

### iOS側の実装メモ

1. 元画像を `prepare_segmentation_dataset.py` と同じ手順(letterbox: 長辺・短辺を512×320に収め、余白は白)で前処理してからモデルに入力する
2. モデル出力(確率マップ)を0.5で二値化
3. letterbox時のスケール・パディング量(`manifest.json`と同じ考え方)を使って、元の撮影画像の座標系に逆変換する
4. 手書きと判定されたピクセルを白で置き換える(`predict_mask.py`の`_cleaned.jpg`と同じロジック)

---

## データを追加するとき

1. 新しい手書き候補クロップを `cropped_images/` に追加
2. `python prepare_segmentation_dataset.py` を再実行(512×320に統一)
3. `python generate_labelstudio_tasks_segmentation.py` → 生成されたJSONのうち**新規分のみ**Label Studioにインポート(既存タスクの重複インポートに注意)
4. アノテーション
5. `python export_masks_from_db.py <プロジェクトID>` でマスクを再生成
6. `python train_segmentation.py` で再学習
