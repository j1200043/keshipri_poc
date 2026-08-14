import cv2
import os
import numpy as np
from ultralytics import YOLO

def remove_handwriting_pixel_level(image, model, conf_threshold=0.15, margin=3):
    """
    YOLOで検出された枠内の「手書き筆跡ピクセル」だけを精度高く判別して白塗り消去する
    ※ モデルオブジェクトを直接受け取るように最適化
    """
    results = model(image, conf=conf_threshold)
    
    cleaned_image = image.copy()
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 枠線まで巻き込まないようマージンを適用
            x1_m = max(0, x1 - margin)
            y1_m = max(0, y1 - margin)
            x2_m = min(w, x2 + margin)
            y2_m = min(h, y2 + margin)
            
            roi_gray = gray[y1_m:y2_m, x1_m:x2_m]
            roi_color = cleaned_image[y1_m:y2_m, x1_m:x2_m]
            
            if roi_gray.size == 0:
                continue

            # --- パーセンタイル閾値による筆跡抽出 ---
            p_75 = np.percentile(roi_gray, 75)  # 領域内の紙地に近い明るさを取得
            pencil_mask = roi_gray < (p_75 - 25)  # 紙地より一定以上暗いピクセルを抽出
            
            # 手書きピクセルを完全な白で置換
            roi_color[pencil_mask] = [255, 255, 255]
                
    return cleaned_image

def apply_clean_binarization(image):
    """
    適応的二値化による仕上げ（ノイズ・かすれ除去機能つき）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ノイズ低減のためのガウシアンフィルタ
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 適応的二値化
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        25,  # ブロックサイズ
        10   # 差分定数
    )
    
    # ポツポツ残る手書き消し残しノイズを除去（オープニング処理）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary_cleaned

if __name__ == "__main__":
    input_dir = "data/images"
    output_dir = "data/output"
    model_path = "runs/detect/models/runs/keshipuri_yolo11s_poc/weights/best.pt"
    
    # モデルファイルの存在チェック
    if not os.path.exists(model_path):
        print(f"【エラー】モデルファイルが見つかりません: {os.path.abspath(model_path)}")
        exit(1)

    # 入力ディレクトリのチェック
    if not os.path.exists(input_dir):
        print(f"【エラー】入力フォルダが見つかりません: {input_dir}")
        exit(1)

    # YOLOモデルを1回だけロード
    print(f"モデルをロードしています: {model_path}")
    model = YOLO(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 処理対象の画像ファイルを一括取得（.jpg, .jpeg, .png）
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"【警告】 {input_dir} 内に処理対象の画像が見つかりませんでした。")
        exit(0)

    print(f"\n合計 {len(image_files)} 件の画像を一括処理します...")
    print("------------------------------------------")

    # フォルダ内の画像をループ処理
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"【スキップ】画像を読み込めませんでした: {filename}")
            continue

        print(f"処理中: {filename} ...", end="", flush=True)

        # 1. 手書きピクセル消去
        step1_img = remove_handwriting_pixel_level(img, model=model, conf_threshold=0.15, margin=3)

        # 2. 白黒二値化仕上げ
        binary_img = apply_clean_binarization(step1_img)

        # 出力ファイル名の生成（例: IMG_7522_cleaned.jpg）
        name_without_ext, ext = os.path.splitext(filename)
        output_filename = f"{name_without_ext}_cleaned{ext}"
        output_path = os.path.join(output_dir, output_filename)

        # 画像保存
        cv2.imwrite(output_path, binary_img)
        print(f" 完了 -> {output_path}")

    print("------------------------------------------")
    print("全画像の処理が完了しました！")