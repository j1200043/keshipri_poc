import os
import sys
import uuid
import django
import numpy as np
import cv2

# data/pseudo_masks/ の疑似マスクを、Label StudioのPrediction(下書き)として
# 既存タスクに読み込ませる。これにより、既に手作業でアノテーション済みのタスクは
# 変更せず、未着手のタスクだけ「ゼロから塗る」のではなく「間違いを直すだけ」に
# 作業を軽減できる。
#
# 使い方: python import_pseudo_predictions.py [プロジェクトID]
# (プロジェクトID省略時は先頭に見つかったプロジェクトを一覧表示して終了)

MASK_DIR = "data/pseudo_masks"
LABEL_NAME = "handwriting"
FROM_NAME = "tag"
TO_NAME = "image"
MODEL_VERSION = "pencil_heuristic_v1"

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


def main():
    setup_django()
    from label_studio_sdk.converter import brush
    from projects.models import Project
    from tasks.models import Prediction

    if len(sys.argv) < 2:
        print("プロジェクト一覧:")
        for p in Project.objects.all():
            print(f"  id={p.id}  title={p.title}  tasks={p.tasks.count()}")
        print("\n使い方: python import_pseudo_predictions.py <プロジェクトID>")
        sys.exit(1)

    project_id = int(sys.argv[1])
    project = Project.objects.get(id=project_id)
    print(f"対象プロジェクト: id={project.id} title={project.title}")

    created = 0
    skipped_annotated = 0
    skipped_no_mask = 0
    skipped_existing_pred = 0

    for task in project.tasks.all():
        filename = image_filename_from_data(task.data)
        if not filename:
            continue

        if task.annotations.exists():
            skipped_annotated += 1
            continue

        if task.predictions.filter(model_version=MODEL_VERSION).exists():
            skipped_existing_pred += 1
            continue

        base_name = os.path.splitext(filename)[0]
        mask_path = os.path.join(MASK_DIR, base_name + ".png")
        if not os.path.exists(mask_path):
            skipped_no_mask += 1
            print(f"【スキップ】マスクなし: {filename}")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]
        mask_binary = (mask > 0).astype(np.uint8) * 255
        rle = brush.mask2rle(mask_binary)

        result = [{
            "id": uuid.uuid4().hex[:10],
            "type": "brushlabels",
            "value": {
                "format": "rle",
                "rle": rle,
                "brushlabels": [LABEL_NAME],
            },
            "origin": "manual",
            "to_name": TO_NAME,
            "from_name": FROM_NAME,
            "image_rotation": 0,
            "original_width": width,
            "original_height": height,
        }]

        Prediction.objects.create(
            task=task,
            project=project,
            result=result,
            model_version=MODEL_VERSION,
            score=0.5,
        )
        created += 1
        print(f"[OK] {filename} -> prediction作成")

    print("--------------------------------------------------")
    print(f"作成: {created} 件")
    print(f"スキップ(既にアノテーション済み): {skipped_annotated} 件")
    print(f"スキップ(疑似マスクなし): {skipped_no_mask} 件")
    print(f"スキップ(既にpredictionあり): {skipped_existing_pred} 件")


if __name__ == "__main__":
    main()
