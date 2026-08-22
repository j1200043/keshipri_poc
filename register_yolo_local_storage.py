import os
import sys
import django

# 【YOLO検出パイプライン用】既存プロジェクト(Keshi_YOLO)に
# data_prepare/images_resized/ をLocal Filesストレージとして登録する。
#
# Label Studioは /data/local-files/?d=... でのファイル配信に、
# 対象ディレクトリを LocalFilesImportStorage として事前登録しておくことを要求する
# (未登録だと画像が読み込めず "issue loading URL" エラーになる)。
# セグメンテーション側(Keshi_Segment)で発生したのと同じ問題を先回りして解消する。
#
# 使い方: python register_yolo_local_storage.py <プロジェクトID>

TARGET_SUBDIR = "data_prepare/images_resized"


def setup_django():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings.label_studio")
    import label_studio

    package_dir = os.path.dirname(label_studio.__file__)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    django.setup()


def main():
    setup_django()
    from projects.models import Project
    from io_storages.localfiles.models import LocalFilesImportStorage

    if len(sys.argv) != 2:
        print("プロジェクト一覧:")
        for p in Project.objects.all():
            print(f"  id={p.id}  title={p.title}  tasks={p.tasks.count()}")
        print("\n使い方: python register_yolo_local_storage.py <プロジェクトID>")
        sys.exit(1)

    project = Project.objects.get(id=int(sys.argv[1]))
    abs_path = os.path.abspath(TARGET_SUBDIR)

    existing = LocalFilesImportStorage.objects.filter(project=project, path=abs_path)
    if existing.exists():
        print(f"既に登録済みです: project={project.title} path={abs_path}")
        return

    storage = LocalFilesImportStorage(project=project, path=abs_path, use_blob_urls=True)
    storage.validate_connection()
    storage.save()
    print(f"登録しました: project={project.title} path={abs_path}")


if __name__ == "__main__":
    main()
