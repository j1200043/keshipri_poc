import os
import sys
from ultralytics import YOLO

def train_keshipuri_model():
    dataset_yaml = os.path.abspath("data/dataset/dataset.yaml")
    
    if not os.path.exists(dataset_yaml):
        print(f"【エラー】 dataset.yaml が見つかりません: {dataset_yaml}")
        sys.exit(1)

    print("==========================================")
    print("YOLOv11s による学習を開始します...")
    print(f"データセット: {dataset_yaml}")
    print("==========================================")

    # 1. YOLOv11s (Small) モデルのロード
    model = YOLO("yolo11s.pt")
#    model = YOLO("yolo11m.pt")

    # 2. デバイス判定 (CUDA > MPS > CPU)
    try:
        import torch
        if torch.cuda.is_available():
            device = 0  # または "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"
        
    print(f"使用デバイス: {device}")

    # 3. 学習実行 (全方向の撮影・薄い鉛筆線に対応するデータ拡張ON)
    results = model.train(
        data=dataset_yaml,
        epochs=100,           
        patience=0,         
        imgsz=1024,
        batch=8,                        # ★Mediumモデル向けに4に調整（VRAM 8GB安全圏
        name="keshipuri_yolo11s_poc",  # 保存先フォルダ名を変更
        project="models/runs",
        device=device,
        # --- 台形補正・正位置画像向け最適化 ---
        degrees=5.0,
        flipud=0.0,
        fliplr=0.0,
        exist_ok=True
    )

    best_model_path = os.path.abspath("models/runs/keshipuri_yolo11s_poc/weights/best.pt")
    print(f"\n学習完了！ 最良モデル保存先: {best_model_path}")

if __name__ == "__main__":
    train_keshipuri_model()