import cv2
import os
import numpy as np
import torch
import easyocr
import easyocr.imgproc as imgproc
from ultralytics import YOLO

_CRAFT_READER = None

def get_craft_reader():
    """文字ストローク検出用のCRAFTモデル(EasyOCR同梱)を遅延ロードして使い回す"""
    global _CRAFT_READER
    if _CRAFT_READER is None:
        _CRAFT_READER = easyocr.Reader(['en'], gpu=True, recognizer=False, verbose=False)
    return _CRAFT_READER

def craft_text_score_map(reader, crop_bgr, canvas_size=768, mag_ratio=2.0):
    """CRAFTで「文字ストロークらしさ」のヒートマップ(0.0〜1.0)をcrop_bgrと同じ解像度で返す。
    罫線やJPEGノイズは文字の形をしていないため低スコアになり、除外しやすくなる。"""
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = img_rgb.shape[:2]
    resized, ratio, _ = imgproc.resize_aspect_ratio(img_rgb, canvas_size, cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    target_h32, target_w32 = resized.shape[:2]
    target_h, target_w = int(h0 * ratio), int(w0 * ratio)

    x = imgproc.normalizeMeanVariance(resized)
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(np.array([x])).to(reader.device)

    with torch.no_grad():
        y, _ = reader.detector(x)
    score_text = y[0, :, :, 0].cpu().numpy()

    # ヒートマップ(canvas半解像度) -> パディング除去 -> crop本来の解像度へ戻す
    score_up = cv2.resize(score_text, (target_w32, target_h32), interpolation=cv2.INTER_LINEAR)
    score_valid = score_up[0:target_h, 0:target_w]
    return cv2.resize(score_valid, (w0, h0), interpolation=cv2.INTER_LINEAR)

def remove_red_marks(image):
    """赤ペン消去"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 35, 40])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([168, 35, 40])
    upper_red2 = np.array([180, 255, 255])
    
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    
    result = image.copy()
    result[red_mask > 0] = [255, 255, 255]
    return result

def debug_process_image(image_path, model, output_debug_dir, conf_threshold=0.15, reader=None):
    if reader is None:
        reader = get_craft_reader()

    img = cv2.imread(image_path)
    if img is None:
        print(f"【エラー】画像の読み込みに失敗: {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    h, w, _ = img.shape

    # --- Step 2: 赤ペン消去 ---
    no_red = remove_red_marks(img)

    # --- Step 3: YOLO推論 & 検出枠の可視化 ---
    results = model(no_red, conf=conf_threshold, augment=True)  # TTA有効化
    
    boxes = []
    box_vis = no_red.copy()
    for r in results:
        for b in r.boxes:
            box = list(map(int, b.xyxy[0]))
            boxes.append(box)
            # 検出枠を緑色の矩形で描画
            cv2.rectangle(box_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(box_vis, f"{float(b.conf[0]):.2f}", (box[0], max(15, box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(output_debug_dir, f"{base_name}_03_yolo_boxes.jpg"), box_vis)

    # --- Step 4: YOLO枠内の手書きピクセル特定 & マスク作成 ---
    gray = cv2.cvtColor(no_red, cv2.COLOR_BGR2GRAY)
    total_pencil_mask = np.zeros((h, w), dtype=np.uint8)
    craft_score_vis = np.zeros((h, w), dtype=np.uint8)      # 4a-1: CRAFT生ヒートマップ
    craft_binary_vis = np.zeros((h, w), dtype=np.uint8)     # 4a-2: 二値化直後（膨張なし）
    craft_mask_vis = np.zeros((h, w), dtype=np.uint8)       # 4a-3: 最終的な文字マスク（細め）
    margin = 2
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for (x1, y1, x2, y2) in boxes:
        x1_m = max(0, x1 - margin)
        y1_m = max(0, y1 - margin)
        x2_m = min(w, x2 + margin)
        y2_m = min(h, y2 + margin)

        roi_gray = gray[y1_m:y2_m, x1_m:x2_m]
        roi_bgr = no_red[y1_m:y2_m, x1_m:x2_m]
        if roi_gray.size == 0:
            continue

        # CRAFT(文字ストローク検出AI)で「文字っぽい形」の領域を特定する
        # （罫線やJPEGノイズは文字の形をしていないため低スコアになり、除外される）
        score = craft_text_score_map(reader, roi_bgr)
        craft_score_vis[y1_m:y2_m, x1_m:x2_m] = (np.clip(score, 0, 1) * 255).astype(np.uint8)

        # 二値化（まだ膨張しない = 生のストローク幅）
        # しきい値0.5: CRAFTのヒートマップはぼやけて広がるため、0.10だと文字全体が
        # 塗りつぶされた太いマスクになってしまう。確信度の高い「芯」だけを残し、細くする
        text_mask_binary = (score > 0.5).astype(np.uint8) * 255
        craft_binary_vis[y1_m:y2_m, x1_m:x2_m] = text_mask_binary

        # 文字マスクは細めに保つため、ごく小さい膨張のみ（ストローク内の小さな穴埋め程度）
        text_mask = cv2.dilate(text_mask_binary, kernel_thin, iterations=1)
        craft_mask_vis[y1_m:y2_m, x1_m:x2_m] = text_mask

        # 純白の紙地だけを除外するゆるいガード（CRAFT領域の取りこぼしを防ぐ）
        loose_dark_mask = (roi_gray < 235).astype(np.uint8) * 255
        local_mask = cv2.bitwise_and(loose_dark_mask, text_mask)

        total_pencil_mask[y1_m:y2_m, x1_m:x2_m] = local_mask

    # CRAFTマスクができるまでの各ステップを保存
    cv2.imwrite(os.path.join(output_debug_dir, f"{base_name}_04a1_craft_score_raw.jpg"), craft_score_vis)
    cv2.imwrite(os.path.join(output_debug_dir, f"{base_name}_04a2_craft_score_binary.jpg"), craft_binary_vis)
    cv2.imwrite(os.path.join(output_debug_dir, f"{base_name}_04a3_craft_textmask.jpg"), craft_mask_vis)

    # 鉛筆判定された部分だけを白黒マスクとして保存（白＝消去対象）
    cv2.imwrite(os.path.join(output_debug_dir, f"{base_name}_04_pencil_mask.jpg"), total_pencil_mask)

    # --- Step 5: 手書きピクセル白塗り ---
    pencil_erased = no_red.copy()
    pencil_erased[total_pencil_mask > 0] = [255, 255, 255]
    cv2.imwrite(os.path.join(output_debug_dir, f"{base_name}_05_pencil_erased.jpg"), pencil_erased)

    # --- Step 6: 影・照明ムラ除去 ---
    erased_gray = cv2.cvtColor(pencil_erased, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(erased_gray, (51, 51), 0)
    flat_gray = cv2.divide(erased_gray, smooth, scale=255)
    cv2.imwrite(os.path.join(output_debug_dir, f"{base_name}_06_flat_gray.jpg"), flat_gray)

    # --- Step 7: 大津の二値化 (Otsu) ---
    blurred = cv2.GaussianBlur(flat_gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(output_debug_dir, f"{base_name}_07_final_otsu.jpg"), binary_cleaned)

    print(f"デバッグ出力完了: {base_name} (03〜07 を保存)")

if __name__ == "__main__":
    input_dir = "data/images"
    output_debug_dir = "data/debug_steps"
    model_path = "runs/detect/models/runs/keshipuri_yolo11s_poc/weights/best.pt"

    if not os.path.exists(model_path):
        print(f"【エラー】モデルが見つかりません: {os.path.abspath(model_path)}")
        exit(1)

    if not os.path.exists(input_dir):
        print(f"【エラー】フォルダが見つかりません: {input_dir}")
        exit(1)

    os.makedirs(output_debug_dir, exist_ok=True)
    model = YOLO(model_path)
    reader = get_craft_reader()

    valid_exts = ('.jpg', '.jpeg', '.png')
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])

    print("==================================================")
    print(f"デバッグ用ステップ出力開始 (全 {len(files)} 枚)")
    print(f"保存先: {os.path.abspath(output_debug_dir)}")
    print("==================================================")

    for f in files:
        debug_process_image(
            os.path.join(input_dir, f), 
            model, 
            output_debug_dir, 
            conf_threshold=0.20,
            reader=reader)

    print("\nすべてのデバッグ画像の生成が完了しました！")
    