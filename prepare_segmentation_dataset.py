import os
import json
import cv2
import numpy as np

# セグメンテーション学習用: cropped_images/ の画像を
# 「向き正規化 + 縮小のみ(拡大しない) + 白パディング」で固定サイズに統一する。
#
# 目標サイズは実データ(90枚)の分布から決定:
#   長辺512 / 短辺320 (32の倍数, median アスペクト比1.6に近い)
#   -> 縮小が必要な画像は全体の1割程度、最大でも1.63倍程度の縮小で収まるためタイル分割は不要。

INPUT_DIR = "data_prepare/cropped_images"
OUTPUT_DIR = "data/images_letterboxed"
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")

TARGET_LONG = 512
TARGET_SHORT = 320
PAD_COLOR = (255, 255, 255)  # 白パディング(紙の背景色に合わせる)


def letterbox(img, target_long=TARGET_LONG, target_short=TARGET_SHORT, pad_color=PAD_COLOR):
    """
    画像を向き正規化(縦長は横長に回転) -> 縮小のみ(拡大しない) -> 白パディング で
    (target_long, target_short) の固定キャンバスに収める。

    戻り値: (canvas, meta)
      meta には元サイズ・回転有無・スケール・パディング量を記録する。
      同じ変換をマスク画像にも適用する際に使う。
    """
    h, w = img.shape[:2]
    rotated = h > w
    if rotated:
        # 縦長 -> 横長に統一(90度回転)。学習時は元々 flipud/fliplr/180度回転を
        # augmentation で使っている前提なので、向きの正規化は精度に悪影響を与えない。
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]

    long_side, short_side = max(w, h), min(w, h)
    scale = min(target_long / long_side, target_short / short_side, 1.0)  # 拡大はしない

    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.full((target_short, target_long, 3), pad_color, dtype=np.uint8)
    pad_left = (target_long - new_w) // 2
    pad_top = (target_short - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    meta = {
        "orig_w": w if not rotated else img.shape[1],
        "orig_h": h if not rotated else img.shape[0],
        "rotated_90cw": rotated,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "resized_w": new_w,
        "resized_h": new_h,
    }
    return canvas, meta


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"【エラー】入力フォルダが見つかりません: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    valid_exts = (".jpg", ".jpeg", ".png")
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_exts))

    if not files:
        print(f"【案内】{INPUT_DIR} に対象画像が見つかりませんでした。")
        return

    print("==================================================")
    print("【セグメンテーション用データセット統一処理】")
    print(f" 入力元    : {INPUT_DIR}")
    print(f" 出力先    : {OUTPUT_DIR}")
    print(f" 目標サイズ: 長辺{TARGET_LONG} x 短辺{TARGET_SHORT}")
    print(f" 対象枚数  : {len(files)} 枚")
    print("==================================================")

    manifest = {}
    shrink_count = 0

    for i, filename in enumerate(files):
        in_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(in_path)
        if img is None:
            print(f"[{i+1}/{len(files)}] 【失敗】 読み込めません: {filename}")
            continue

        canvas, meta = letterbox(img)
        out_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(out_path, canvas)
        manifest[filename] = meta

        if meta["scale"] < 1.0:
            shrink_count += 1
            print(f"[{i+1}/{len(files)}] {filename} : 縮小 scale={meta['scale']:.3f} "
                  f"({meta['orig_w']}x{meta['orig_h']} -> {meta['resized_w']}x{meta['resized_h']})")
        else:
            print(f"[{i+1}/{len(files)}] {filename} : 等倍(パディングのみ)")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("--------------------------------------------------")
    print(f"完了しました！ (処理枚数: {len(manifest)}, 縮小あり: {shrink_count})")
    print(f"manifest保存先: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
