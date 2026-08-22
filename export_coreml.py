import os
from ultralytics import YOLO

# 1. 実際に生成された best.pt のパスを指定
model_path = os.path.abspath("runs/detect/models/runs/keshipuri_yolo11s_poc/weights/best.pt")

if not os.path.exists(model_path):
    print(f"【エラー】ファイルが見つかりません: {model_path}")
else:
    print(f"モデルをロードしています: {model_path}")
    model = YOLO(model_path)
    
    print("CoreML (.mlpackage) への変換を開始します...")
    # iOS向けに最適化してエクスポート
    model.export(
        format="coreml",
        imgsz=1024,
        half=False,      # FP32で安定化（シミュレータでのFP16非互換を回避）
        dynamic=False    # 固定シェイプ化（MLIR pass manager failed の主要因を排除）
    )
    print("CoreML 変換完了！ runs/detect/models/runs/keshipuri_yolo_poc/weights/ 配下に保存されました。")