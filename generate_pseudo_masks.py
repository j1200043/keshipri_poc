import os
import json
import cv2
import numpy as np

# data/images_letterboxed/ の各画像に対して、鉛筆書き(暗いピクセル)を粗く検出した
# 疑似マスクを自動生成する。人手で全部塗るのではなく、これをLabel Studioの
# Predictions(下書き)として読み込み、"修正するだけ"にして作業量を減らすのが目的。
#
# 精度は完璧ではない(黒っぽい印刷罫線なども拾ってしまうことがある)前提で、
# あくまで手作業の叩き台として使う。

IMAGE_DIR = "data/images_letterboxed"
MANIFEST_PATH = os.path.join(IMAGE_DIR, "manifest.json")
MASK_DIR = "data/pseudo_masks"
PREVIEW_DIR = "data/pseudo_masks_preview"


def make_mask(img_bgr, pad_left, pad_top, resized_w, resized_h):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 白パディング部分は最初からマスク対象外にする(手書きが存在し得ないため)
    content = gray[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w]

    # Otsuの二値化で「暗いピクセル(鉛筆・ペン)」を抽出
    _, dark = cv2.threshold(content, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 罫線のドットなど孤立した小さいノイズを除去(モルフォロジー・オープニング)
    kernel = np.ones((2, 2), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    # 印刷された罫線・枠線(=長くまっすぐな線)をHough変換で検出して除外する。
    # 手書き線は直線検出のmin長さに達しにくい/曲がっているため残りやすい。
    edges = cv2.Canny(content, 50, 150)
    min_len = int(min(resized_w, resized_h) * 0.35)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=4)
    if lines is not None:
        line_mask = np.zeros_like(dark)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=4)
        dark = cv2.bitwise_and(dark, cv2.bitwise_not(line_mask))

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = dark
    return mask


def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"【エラー】manifestが見つかりません: {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    count = 0
    for filename, meta in sorted(manifest.items()):
        img_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"【スキップ】読み込めません: {filename}")
            continue

        mask = make_mask(img, meta["pad_left"], meta["pad_top"], meta["resized_w"], meta["resized_h"])

        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(MASK_DIR, base_name + ".png"), mask)

        overlay = img.copy()
        overlay[mask > 0] = (0, 0, 255)  # 赤で重ねて可視化(BGR)
        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        cv2.imwrite(os.path.join(PREVIEW_DIR, base_name + ".jpg"), blended)

        count += 1

    print(f"完了: {count}枚の疑似マスクを生成しました。")
    print(f"マスク: {os.path.abspath(MASK_DIR)}")
    print(f"プレビュー(赤重ね): {os.path.abspath(PREVIEW_DIR)}")


if __name__ == "__main__":
    main()
