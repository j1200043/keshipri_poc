import os
import json

# 【YOLO検出パイプライン用】Label Studioにインポートするタスクを生成する。
# セグメンテーション用(generate_labelstudio_tasks_segmentation.py)とは別プロジェクト・
# 別ラベリング設定(RectangleLabels)なので混同しないこと。
#
# data_prepare/images_resized/ (台形補正・リサイズ済みの元画像)を対象にする。
# 新しく画像を追加してYOLOアノテーションしたい場合は、このフォルダに画像を追加してから
# 本スクリプトを再実行し、生成されたJSONをLabel Studioの [Import] でアップロードする。
#
# 前提: Label Studio をローカルファイル配信モードで起動していること
#   LOCAL_FILES_SERVING_ENABLED=true LOCAL_FILES_DOCUMENT_ROOT=$(pwd) label-studio start
# かつ、対象プロジェクトの Settings > Cloud Storage で
#   Local files / Absolute local path = <このリポジトリの絶対パス>/data_prepare/images_resized
# を追加済みであること(register_yolo_local_storage.py で登録済み)。

IMAGE_DIR = "data_prepare/images_resized"
OUTPUT_PATH = "data/labelstudio_tasks_yolo.json"


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
    print("※既存45枚は手動アップロード済みのタスクとして既に存在するため、")
    print("  同じプロジェクトにこのJSONをインポートすると重複します。新規追加分だけに使ってください。")


if __name__ == "__main__":
    main()
