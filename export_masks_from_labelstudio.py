import os
import sys
import json
import numpy as np
from PIL import Image
from label_studio_sdk.converter import brush

# Label Studioでエクスポートした JSON(Brushアノテーション, RLE形式)を
# data/images_letterboxed/ の各画像と同名の二値マスクPNGに変換する。
#
# 使い方:
#   1. Label Studio上で Export -> JSON を選び、書き出したファイルを指定する
#   2. python export_masks_from_labelstudio.py <エクスポートJSONのパス>
#
# 出力: data/masks/<画像名>.png (白=手書きピクセル, 黒=背景)
# 画像は data/images_letterboxed/ で既に512x320に統一済みのものをそのまま
# アノテーションしている前提なので、追加の座標変換は不要。

OUTPUT_DIR = "data/masks"
TARGET_LABEL = "handwriting"


def rle_to_mask(result):
    rle = result["value"]["rle"]
    width = result["original_width"]
    height = result["original_height"]
    decoded = brush.decode_rle(rle)
    # decode_rle は [width, height, 4] を flatten した配列を返す(RGBA、alphaが塗り部分)
    arr = decoded.reshape((height, width, 4))
    alpha = arr[:, :, 3]
    return alpha > 0


def image_filename_from_data(data):
    # 例: "/data/local-files/?d=data/images_letterboxed/IMG_7520_001.jpg"
    image_ref = data.get("image", "")
    return os.path.basename(image_ref.split("?d=")[-1])


def main():
    if len(sys.argv) != 2:
        print("使い方: python export_masks_from_labelstudio.py <エクスポートJSONのパス>")
        sys.exit(1)

    export_path = sys.argv[1]
    if not os.path.exists(export_path):
        print(f"【エラー】ファイルが見つかりません: {export_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(export_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    saved = 0
    skipped = 0
    for task in tasks:
        filename = image_filename_from_data(task.get("data", {}))
        if not filename:
            continue

        annotations = task.get("annotations", [])
        if not annotations:
            skipped += 1
            print(f"【スキップ】アノテーションなし: {filename}")
            continue

        # 複数人アノテーションがある場合は最初のものを採用
        results = annotations[0].get("result", [])
        mask = None
        for result in results:
            if result.get("type") != "brushlabels":
                continue
            if TARGET_LABEL not in result.get("value", {}).get("brushlabels", []):
                continue
            region_mask = rle_to_mask(result)
            mask = region_mask if mask is None else (mask | region_mask)

        if mask is None:
            skipped += 1
            print(f"【スキップ】'{TARGET_LABEL}' ラベルの領域なし: {filename}")
            continue

        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, base_name + ".png")
        mask_img.save(out_path)
        saved += 1
        print(f"[OK] {filename} -> {out_path}")

    print("--------------------------------------------------")
    print(f"完了: 保存 {saved} 件 / スキップ {skipped} 件")
    print(f"保存先: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
