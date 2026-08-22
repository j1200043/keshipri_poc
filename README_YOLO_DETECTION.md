# 『消しプリ』手書き領域検出パイプライン(YOLO)

「撮影したプリントから手書き文字を自動消去する」機能のうち、**手書きがどこにあるかを矩形(バウンディングボックス)で検出する**パイプラインです。

⚠️ [README_SEGMENTATION.md](README_SEGMENTATION.md)(検出領域内の手書き**ピクセル**を判定するセグメンテーションパイプライン)とは別物です。混同しないよう、スクリプト名・Label Studioプロジェクト・出力先をすべて分けています。両者の対応関係は [README_SEGMENTATION.md](README_SEGMENTATION.md) 冒頭の表を参照してください。

---

## 1. プロジェクト構造(最終構成)

```text
keshipri_poc/
├── data/
│   ├── images/               # 元のテストプリント画像
│   ├── images_resized/       # Label Studio用リサイズ画像
│   ├── dataset/              # Label Studioからエクスポートしたデータ
│   │   ├── dataset.yaml      # データセット定義ファイル
│   │   ├── images/           # アノテーション済み画像
│   │   └── labels/           # YOLOフォーマットのテキストラベル
│   └── output/               # パイプライン実行結果の出力先
├── models/
│   └── runs/                 # 学習済みモデル(best.pt)の保存先
├── resize_for_annotation.py  # 画像事前リサイズスクリプト
├── train.py                  # YOLOv11s 学習スクリプト
├── export_coreml.py          # iOS用 CoreML 変換スクリプト
└── infer_and_clean.py        # 推論・手書きピクセル消去・二値化実行スクリプト
```

---

## 2. 環境構築手順

### Step 1: Python仮想環境の作成とライブラリ一括インストール

```bash
mkdir -p keshipri_poc
cd keshipri_poc

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -U ultralytics opencv-python label-studio coremltools pillow torch
```

---

## 3. アノテーション作業(Label Studio)

Label Studioプロジェクト名: **`Keshi_YOLO`**(RectangleLabels、ラベル名 `handwriting`)

### Step 1: 画像の軽量化(ブラウザ描画エラー防止)

```python
# resize_for_annotation.py
import os
from PIL import Image

def resize_images(input_dir="data/images", output_dir="data/images_resized", max_size=1280):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        with Image.open(input_path) as img:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=85)

if __name__ == "__main__":
    resize_images()
```

(既存の `batch_resize.py` も同様の用途で `data_prepare/images_transformed` → `data_prepare/images_resized` のリサイズに使われている)

### Step 2: Label Studio の起動とプロジェクト設定

```bash
LOCAL_FILES_SERVING_ENABLED=true LOCAL_FILES_DOCUMENT_ROOT=$(pwd) label-studio start
```

1. ブラウザ(`http://localhost:8080`)で **Create** ➔ **Labeling Setup** ➔ Custom template → `label_studio_config_yolo.xml` の内容を貼り付け
2. `data_prepare/images_resized/` 内の画像をインポートし、手書き部分(式、数字、記号)をまとまり(`= 6` など)で四角で囲む
3. 作業後、**Export** ➔ **YOLO** を選択して Zip をダウンロード

### Step 2': ローカルファイルから直接アノテーションする(新規追加分向け)

初期の45枚は手動アップロードで作成済みだが、以降**新しく画像を追加する場合**はローカルファイル参照でインポートできる環境を整えてある。

```bash
# 1. 対象プロジェクトにLocal Filesストレージを登録(初回のみ)
python register_yolo_local_storage.py <プロジェクトID>

# 2. data_prepare/images_resized/ に新しい画像を追加してから、タスクJSONを生成
python generate_labelstudio_tasks_yolo.py

# 3. 生成された data/labelstudio_tasks_yolo.json をLabel StudioのUIで [Import]
#    (既存タスクと重複しないよう、新規追加分のみが対象になっていることを確認する)
```

### Step 3: データセットの配置と `dataset.yaml` の作成

1. Zipを解凍し、中身を `data/dataset/` に配置
2. `data/dataset/dataset.yaml` を新規作成:
```yaml
path: data/dataset
train: images
val: images

names:
  0: handwriting
```

---

## 4. AIモデルの学習(YOLOv11s)

`train.py`:

```python
import os
import sys
from ultralytics import YOLO

def train_keshipuri_model():
    dataset_yaml = os.path.abspath("data/dataset/dataset.yaml")
    if not os.path.exists(dataset_yaml):
        print(f"【エラー】 dataset.yaml が見つかりません: {dataset_yaml}")
        sys.exit(1)

    print("YOLOv11s による学習を開始します...")
    model = YOLO("yolo11s.pt")

    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=8,
        name="keshipuri_yolo11s_poc",
        project="models/runs",
        device=device,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        exist_ok=True
    )
    print(f"学習完了！ モデル保存先: models/runs/keshipuri_yolo11s_poc/weights/best.pt")

if __name__ == "__main__":
    train_keshipuri_model()
```

実行コマンド:
```bash
python train.py
```

---

## 5. 推論 ＆ ピクセルレベル消去パイプラインの実行

枠線や印刷文字を残し、「手書きの暗いピクセルだけ」を白塗りで消去するパイプライン。中身は `infer_and_clean.py` を参照。

実行コマンド:
```bash
python infer_and_clean.py
```

> 補足: このヒューリスティック消去(ROI内の局所平均より暗いピクセルを白塗り)を、より精度良く行うのが [README_SEGMENTATION.md](README_SEGMENTATION.md) のセグメンテーションモデルの役割。

---

## 6. iOS組み込み用 CoreML への変換

`export_coreml.py`:

```python
import os
from ultralytics import YOLO

model_path = os.path.abspath("models/runs/keshipuri_yolo11s_poc/weights/best.pt")
if os.path.exists(model_path):
    model = YOLO(model_path)
    model.export(format="coreml", nms=True, imgsz=640)
    print("CoreML 変換成功！")
```

実行コマンド:
```bash
python export_coreml.py
```

---

## 💡 精度向上と運用のノウハウ

- **データ量:** 目安として **30〜50枚** のアノテーションで実用的な精度に達する
- **アノテーションのコツ:** 単体文字ではなく、`= 6` や `（1）` のように **手書きの固まり・文脈ごと大きめに囲む** と検知率が上がる(セグメンテーション側とは逆に、印刷部分を含めて囲んでよい)
- **手書きピクセル消去:** `roi_gray < (mean_val - 15)` の `-15` の値を調整することで、消去の強さを調整可能
- データセット全体における背景画像の比率は 全体の 5%〜10% 程度(多くても15%以下)が目安
- 台形補正されたデータを使って学習する
- Precision: モデルが検出したものに対して、正解だった割合
- Recall: 正解枠の総数に対して検出できたものの割合
- mAP50(Box mAP@0.5): 検出枠の総合精度(緩めの基準)
- mAP50-95 (Box mAP@0.5:0.95): 検出枠の総合精度(厳格な基準)

yolo11sの結果。総数 34枚。5枚が背景。

| ターミナル表記 | 正式名称 | 今回の値 | 意味 |
|---|---|---|---|
| Box(P | Box Precision | 0.958 (95.8%) | 検出枠の適合率(検出した枠の正確さ) |
| R | Box Recall | 0.850 (85.0%) | 検出枠の再現率(見落としの少なさ) |
| mAP50 | Box mAP@0.5 | 0.950 (95.0%) | 検出枠の総合精度(緩めの基準) |
| mAP50-95 | Box mAP@0.5:0.95 | 0.629 (62.9%) | 検出枠の総合精度(厳格な基準) |

- 本番投入の目安
    - BoxP: 85%以上
    - R: 85~90%以上
    - mAP50: 85%〜90%以上
    - mAP50-95: 60%以上

今のままでも十分本番投入できるレベルではあるが、もう少し精度を上げていきたい。競合はもっとレベルが高いから。

- train中は、confidenceは0.25である。数値を上げるとPが上昇する。絶対にミスしない。ただし、抜け漏れが生じる。プリント修正の場合は、confを少し下げてでも消し漏れを防ぎたいので、confを0.20とかに下げてやってみるといい。

## 精度を上げるためにできそうなこと

- yolo11mにする。学習データが30-40枚だと逆に精度がおちることがわかった。200-500枚ならば、yolo11mを検討する。
- TTA(Test Time Augmentation)の有効化: 推論時に画像を反転・拡大縮小して複数回判定し、結果を統合することで見落とし(Recall)を大幅に削減する。
```
results = model.predict(source="test.jpg", augment=True)  # TTA有効化
```
