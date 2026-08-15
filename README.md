失礼いたしました！内部のコードブロック記述が重なって途中で表示が切れてしまったようです。

全文を途切れることなくそのままコピー＆ペーストできるように、**外側の枠を標準のプレーンテキスト形式にして、中身のMarkdownコードを1つにまとめました。**

以下のテキストをすべてコピーし、`PoC_MANUAL.md` や `README.md` といったファイル名で保存してください！

==================================================

# 『消しプリ』AIモデル検証（PoC）再現マニュアル

「撮影したプリントから手書き文字を自動消去する」機能を再現・再検証するための環境構築およびパイプライン構築マニュアルです。

---

## 1. プロジェクト構造（最終構成）

以下のようなフォルダ・ファイル構成を作成します。

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
│   └── runs/                 # 学習済みモデル（best.pt）の保存先
├── resize_for_annotation.py  # 画像事前リサイズスクリプト
├── train.py                  # YOLOv11s 学習スクリプト
├── export_coreml.py          # iOS用 CoreML 変換スクリプト
└── infer_and_clean.py        # 推論・手書きピクセル消去・二値化実行スクリプト

```

---

## 2. 環境構築手順

### Step 1: Python仮想環境の作成とライブラリ一括インストール

```bash
# プロジェクトフォルダの作成
mkdir -p keshipri_poc
cd keshipri_poc

# Python 仮想環境の作成と有効化
python3 -m venv venv
source venv/bin/activate

# 必要なライブラリの一括インストール・最新化
python -m pip install --upgrade pip
pip install -U ultralytics opencv-python label-studio coremltools pillow torch

```

---

## 3. アノテーション作業（Label Studio）

### Step 1: 画像の軽量化（ブラウザ描画エラー防止）

大きな撮影画像（数MB）をそのまま読むと Label Studio が重くなるため、事前にリサイズします。

`resize_for_annotation.py`:

```python
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

### Step 2: Label Studio の起動とプロジェクト設定

1. **ターミナルで起動:**
```bash
LOCAL_FILES_SERVING_ENABLED=true label-studio

```


2. ブラウザ（`http://localhost:8080`）で **Create** ➔ **Labeling Setup** ➔ **Computer Vision** ➔ **Object Detection with Bounding Boxes** を選択。
3. ラベル名に `handwriting` を追加して **Save**。
4. `data/images_resized/` 内の画像をインポートし、手書き部分（式、数字、記号）をまとまり（`= 6` など）で四角で囲む。
5. 作業後、**Export** ➔ **YOLO** を選択して Zip をダウンロード。

### Step 3: データセットの配置と `dataset.yaml` の作成

1. Zipを解凍し、中身を `data/dataset/` に配置。
2. `data/dataset/dataset.yaml` を新規作成：
```yaml
path: data/dataset
train: images
val: images

names:
  0: handwriting

```



---

## 4. AIモデルの学習（YOLOv11s）

全角度対応（回転・上下左右反転）を自動適用した `train.py` を作成して実行します。

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
        epochs=100,            # データ量に応じて100〜150に変更
        imgsz=640,
        batch=8,
        name="keshipuri_yolo11s_poc",
        project="models/runs",
        device=device,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=180.0,         # 全方向の回転をカバー
        flipud=0.5,            # 上下反転
        fliplr=0.5,            # 左右反転
        exist_ok=True
    )
    print(f"学習完了！ モデル保存先: models/runs/keshipuri_yolo11s_poc/weights/best.pt")

if __name__ == "__main__":
    train_keshipuri_model()

```

実行コマンド：

```bash
python train.py

```

---

## 5. 推論 ＆ ピクセルレベル消去パイプラインの実行

枠線や印刷文字を残し、「手書きの暗いピクセルだけ」を白塗りで消去するパイプラインコードです。

`infer_and_clean.py`:

```python
import cv2
import os
import numpy as np
from ultralytics import YOLO

def remove_handwriting_pixel_level(image, model_path, conf_threshold=0.15, margin=5):
    """
    YOLOで検出された枠内の「暗い筆跡ピクセル」だけをピンポイントで白塗り
    """
    if not os.path.exists(model_path):
        print(f"【エラー】モデルファイルが見つかりません: {model_path}")
        return image

    model = YOLO(model_path)
    results = model(image, conf=conf_threshold)
    
    cleaned_image = image.copy()
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # マージン適用
            x1_m = max(0, x1 - margin)
            y1_m = max(0, y1 - margin)
            x2_m = min(w, x2 + margin)
            y2_m = min(h, y2 + margin)
            
            roi_gray = gray[y1_m:y2_m, x1_m:x2_m]
            roi_color = cleaned_image[y1_m:y2_m, x1_m:x2_m]
            
            if roi_gray.size == 0:
                continue

            # 局所的な平均明るさより暗いピクセル（鉛筆筆跡）を抽出
            mean_val = np.mean(roi_gray)
            pencil_mask = roi_gray < (mean_val - 15)
            
            # 手書きピクセルのみ白置換
            roi_color[pencil_mask] = [255, 255, 255]
                
    return cleaned_image

def apply_clean_binarization(image):
    """
    適応的二値化による白地ベースの仕上げ
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    return binary

if __name__ == "__main__":
    img_path = "data/images/IMG_7520.jpg"  # テスト画像パス
    model_path = "models/runs/keshipuri_yolo11s_poc/weights/best.pt"
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"画像が読み込めません: {img_path}")
        exit(1)
        
    # 1. ピクセル単位で手書き消去
    step1_img = remove_handwriting_pixel_level(img, model_path=model_path, conf_threshold=0.15)

    # 2. 白黒仕上げ
    binary_img = apply_clean_binarization(step1_img)
    
    os.makedirs("data/output", exist_ok=True)
    cv2.imwrite("data/output/cleaned_result.jpg", binary_img)
    print("完了: data/output/cleaned_result.jpg")

```

実行コマンド：

```bash
python infer_and_clean.py

```

---

## 6. iOS組み込み用 CoreML への変換（最終ステップ）

作成した `best.pt` を iPhone アプリ（Swift）に組み込むための `.mlpackage` に変換します。

`export_coreml.py`:

```python
import os
from ultralytics import YOLO

model_path = os.path.abspath("models/runs/keshipuri_yolo11s_poc/weights/best.pt")
if os.path.exists(model_path):
    model = YOLO(model_path)
    # nms=True で重複ボックスの除去処理を組み込み、640pxサイズで書き出し
    model.export(format="coreml", nms=True, imgsz=640)
    print("CoreML 変換成功！")

```

実行コマンド：

```bash
python export_coreml.py

```

---

## 💡 精度向上と運用のノウハウ

* **データ量:** 目安として **30〜50枚** のアノテーションで実用的な精度に達します。
* **アノテーションのコツ:** 単体文字ではなく、`= 6` や `（1）` のように **手書きの固まり・文脈ごと大きめに囲む** と YOLO の検知率が上がります。
* **手書きピクセル消去:** `roi_gray < (mean_val - 15)` の `-15` の値を調整することで、消去の強さ（薄い線を消すか/印刷線を残すか）をチューニング可能です。
- データセット全体における背景画像の比率は 全体の 5%〜10% 程度（多くても15%以下）が目安
- 台形補正されたデータを使って学習する。
- Precisionは、モデルが検出したものに対して、正解だった割合。
- Recallは、正解枠の総数に対して検出できたものの割合。
- mAP50(Box mAP@0.5): 検出枠の総合精度（緩めの基準）
- mAP50-95 (Box mAP@0.5:0.95) : 検出枠の総合精度（厳格な基準）

yolo11sの結果。総数 34枚。5枚が背景。

|ターミナル表記 | 正式名称 | 今回の値 | 意味 | 
|---|---|---|---|
|Box(P     | Box Precision    | 0.958 (95.8%) | 検出枠の適合率（検出した枠の正確さ）|
|R         | Box Recall       | 0.850 (85.0%) | 検出枠の再現率（見落としの少なさ）  |
|mAP50     | Box mAP@0.5      | 0.950 (95.0%) | 検出枠の総合精度（緩めの基準）|
|mAP50-95) | Box mAP@0.5:0.95 | 0.629 (62.9%) | 検出枠の総合精度（厳格な基準）|

- 本番投入の目安
    - BoxP     : 85%以上
    - R        : 85~90%以上
    - mAP50    : 85% 〜 90% 以上
    - mAP50-95 : 60%以上
今のままでも十分本番投入できるレベルではあるが、もう少し精度を上げていきたい。競合はもっとレベルが高いから。

- train中は、confidenceは0.25である。数値を上げるとPが上昇する。絶対にミスしない。ただし、抜け漏れが生じる。プリント修正の場合は、confを少し下げてでも消し漏れを防ぎたいので、confを0.20とかに下げてやってみるといい。

## 精度を上げるためにできそうなこと
- yolo11mにする。学習データが30-40枚だと逆に精度がおちることがわかった。200-500枚ならば、yolo11mを検討する。

- TTA（Test Time Augmentation）の有効化
推論時に画像を反転・拡大縮小して複数回判定し、結果を統合することで見落とし（Recall）を大幅に削減します。
```
results = model.predict(source="test.jpg", augment=True)  # TTA有効化
```
