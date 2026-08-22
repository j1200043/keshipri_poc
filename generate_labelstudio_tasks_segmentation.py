import os
import json

# Label Studio にインポートするタスクJSON(ローカルファイル参照)を生成する。
#
# data/images_letterboxed/ (512x320に統一済みの画像)を対象にする。
# ここでアノテーションしたブラシマスクは、そのまま学習用の画像サイズと
# ピクセル単位で一致するので、後段でマスクを変換し直す必要がない。
#
# 前提: Label Studio をローカルファイル配信モードで起動していること
#   LOCAL_FILES_SERVING_ENABLED=true LOCAL_FILES_DOCUMENT_ROOT=$(pwd) label-studio start

IMAGE_DIR = "data/images_letterboxed"
OUTPUT_PATH = "data/labelstudio_tasks_segmentation.json"


def main():
    valid_exts = (".jpg", ".jpeg", ".png")
    files = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_exts))
    if not files:
        print(f"【エラー】{IMAGE_DIR} に画像が見つかりません。")
        return

    tasks = []
    for filename in files:
        rel_path = os.path.join(IMAGE_DIR, filename)
        tasks.append({"data": {"image": f"/data/local-files/?d={rel_path}"}})

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"タスクJSONを作成しました: {OUTPUT_PATH} ({len(tasks)}件)")
    print("Label StudioのUIで [Import] -> このJSONファイルをアップロードしてください。")


if __name__ == "__main__":
    main()
