import cv2
import os
import numpy as np

# グローバル状態管理
clicked_points = []
scale_ratio = 1.0
pad_margin = 30     # 左右・下の余白マージン(px)
header_height = 40  # 上部黒バーの高さ(px)
img_orig_w = 0
img_orig_h = 0

def mouse_callback(event, x, y, flags, param):
    global clicked_points, scale_ratio, pad_margin, header_height, img_orig_w, img_orig_h
    
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        # ヘッダーバーとパディングを正確に差し引く（ズレの解消）
        img_x = x - pad_margin
        img_y = y - (pad_margin + header_height)
        
        orig_x = int(round(img_x / scale_ratio))
        orig_y = int(round(img_y / scale_ratio))
        
        # 画像外をクリックしても四隅の端に収まるように安全制限
        orig_x = max(0, min(img_orig_w - 1, orig_x))
        orig_y = max(0, min(img_orig_h - 1, orig_y))
        
        clicked_points.append((orig_x, orig_y))

def apply_perspective_transform(image, pts):
    """4点座標をもとに長方形へ射影変換（台形補正）"""
    pts = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = pts

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = max(int(height_a), int(height_b))

    if max_w == 0 or max_h == 0:
        return image

    dst = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_w, max_h))
    return warped

def run_manual_transform(input_dir="data/images", output_dir="data/images_transformed", 
                         max_display_w=950, max_display_h=700):
    global clicked_points, scale_ratio, pad_margin, header_height, img_orig_w, img_orig_h
    
    if not os.path.exists(input_dir):
        print(f"【エラー】フォルダが見つかりません: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.jpg', '.jpeg', '.png')
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
    
    if not files:
        print(f"【案内】{input_dir} に画像ファイルがありません。")
        return

    window_name = "Perspective Transform (Click 4 Corners)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 30, 30)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("==================================================")
    print("【全画像 4点指定台形補正ツール】")
    print(f" 対象枚数: {len(files)} 枚 (座標補正済み)")
    print("--------------------------------------------------")
    print(" [クリック] : 四隅を『左上 -> 右上 -> 右下 -> 左下』の順に指定")
    print(" [u] キー   : 直前の1点を取り消す（Undo）")
    print(" [r] キー   : 4点をすべてリセットしてやり直し")
    print(" [s] キー   : 補正せず元画像をそのまま保存して次へ（Skip）")
    print(" [q] / [ESC]: 作業を中断して終了")
    print("==================================================")

    labels = ["1. Top-Left", "2. Top-Right", "3. Bottom-Right", "4. Bottom-Left"]

    for i, filename in enumerate(files):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        
        img = cv2.imread(in_path)
        if img is None:
            continue

        img_orig_h, img_orig_w = img.shape[:2]
        
        # 画面内に確実に収めるスケール計算
        scale_ratio = min((max_display_w - pad_margin * 2) / img_orig_w, 
                          (max_display_h - pad_margin * 2 - header_height) / img_orig_h, 
                          1.0)
        disp_w, disp_h = int(img_orig_w * scale_ratio), int(img_orig_h * scale_ratio)
        resized_img = cv2.resize(img, (disp_w, disp_h))

        # ベースキャンバス
        canvas_w = disp_w + pad_margin * 2
        canvas_h = disp_h + pad_margin * 2 + header_height
        base_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        base_canvas[:] = (35, 35, 35)

        # 画像の配置
        y_offset = pad_margin + header_height
        base_canvas[y_offset : y_offset + disp_h, pad_margin : pad_margin + disp_w] = resized_img

        clicked_points = []
        
        while True:
            display = base_canvas.copy()
            pts_count = len(clicked_points)

            # クリックした点の描画 & 補助線の描画
            disp_pts = []
            for p in clicked_points:
                cx = int(round(p[0] * scale_ratio)) + pad_margin
                cy = int(round(p[1] * scale_ratio)) + pad_margin + header_height
                disp_pts.append((cx, cy))
            
            for idx, pt in enumerate(disp_pts):
                cv2.circle(display, pt, 6, (0, 0, 255), -1)
                cv2.putText(display, str(idx + 1), (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if pts_count > 1:
                for j in range(pts_count - 1):
                    cv2.line(display, disp_pts[j], disp_pts[j + 1], (0, 255, 0), 2)
            if pts_count == 4:
                cv2.line(display, disp_pts[3], disp_pts[0], (0, 255, 0), 2)

            # ガイドバー
            next_target = labels[pts_count] if pts_count < 4 else "Processing..."
            info_text = f"[{i+1}/{len(files)}] {filename} | Next: {next_target}"
            
            cv2.rectangle(display, (0, 0), (canvas_w, header_height), (20, 20, 20), -1)
            cv2.putText(display, info_text, (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow(window_name, display)

            if pts_count == 4:
                cv2.waitKey(200)
                warped = apply_perspective_transform(img, clicked_points)
                cv2.imwrite(out_path, warped)
                print(f"({i+1}/{len(files)}) 補正完了: {filename}")
                break

            key = cv2.waitKey(25) & 0xFF
            
            if key == ord('u'):  # 取消
                if clicked_points:
                    clicked_points.pop()
            elif key == ord('r'):  # リセット
                clicked_points = []
            elif key == ord('s'):  # スキップ
                cv2.imwrite(out_path, img)
                print(f"({i+1}/{len(files)}) スキップ: {filename}")
                break
            elif key == ord('q') or key == 27:  # 終了
                cv2.destroyAllWindows()
                print("\n途中で終了しました。")
                return

    cv2.destroyAllWindows()
    print("\nすべての画像の台形補正が完了しました！")
    print(f"保存先フォルダ: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    run_manual_transform()