import os
import sys
import django
import numpy as np
from PIL import Image

# Label StudioのDBから直接、完了済みアノテーション(Brushラベル)を読み出して
# data/images_letterboxed/ と同名の二値マスクPNGを data/masks/ に書き出す。
# (JSONを手動でExportする代わりに、DBから直接生成する版)
#
# 使い方: python export_masks_from_db.py <プロジェクトID>

OUTPUT_DIR = "data/masks"
TARGET_LABEL = "handwriting"


def setup_django():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")
    import label_studio

    package_dir = os.path.dirname(label_studio.__file__)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    django.setup()


def image_filename_from_data(data):
    image_ref = data.get("image", "")
    return os.path.basename(image_ref.split("?d=")[-1])


def result_to_mask(result):
    from label_studio_sdk.converter import brush

    width = result["original_width"]
    height = result["original_height"]
    rle = result["value"]["rle"]
    decoded = brush.decode_rle(rle)
    arr = decoded.reshape((height, width, 4))
    alpha = arr[:, :, 3]
    return alpha > 0


def main():
    setup_django()
    from projects.models import Project

    if len(sys.argv) != 2:
        print("使い方: python export_masks_from_db.py <プロジェクトID>")
        sys.exit(1)

    project = Project.objects.get(id=int(sys.argv[1]))
    print(f"対象プロジェクト: id={project.id} title={project.title}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved = 0
    skipped = 0
    for task in project.tasks.all():
        filename = image_filename_from_data(task.data)
        if not filename:
            continue

        annotations = list(task.annotations.all())
        if not annotations:
            skipped += 1
            print(f"【スキップ】アノテーションなし: {filename}")
            continue

        # 複数ある場合は最後(最新)のものを採用
        results = annotations[-1].result or []

        mask = None
        for result in results:
            if result.get("type") != "brushlabels":
                continue
            if TARGET_LABEL not in result.get("value", {}).get("brushlabels", []):
                continue
            region_mask = result_to_mask(result)
            mask = region_mask if mask is None else (mask | region_mask)

        if mask is None:
            # 「手書きなし」の正しい正解として全面0のマスクを保存する(キャンバスは512x320固定)
            mask = np.zeros((320, 512), dtype=bool)

        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, base_name + ".png")
        mask_img.save(out_path)
        saved += 1

    print("--------------------------------------------------")
    print(f"完了: 保存 {saved} 件 / スキップ {skipped} 件")
    print(f"保存先: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
