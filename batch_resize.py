import os
from PIL import Image, ImageOps

def resize_images(input_dir="data/images_transformed", 
                  output_dir="data/images_resized", 
                  max_size=1024, 
                  quality=90):
    """
    指定フォルダ内の画像を一括で長辺指定サイズにリサイズして保存する
    
    :param input_dir: 元画像フォルダのパス
    :param output_dir: リサイズ後画像の保存先フォルダ
    :param max_size: 長辺の最大ピクセル数（デフォルト: 1024）
    :param quality: JPEG保存時の画質（1〜100、デフォルト: 90）
    """
    if not os.path.exists(input_dir):
        print(f"【エラー】入力フォルダが見つかりません: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])

    if not files:
        print(f"【案内】{input_dir} に対象画像が見つかりませんでした。")
        return

    print("==================================================")
    print(f"【一括リサイズ開始】")
    print(f" 入力元  : {input_dir}")
    print(f" 出力先  : {output_dir}")
    print(f" 長辺設定: {max_size} px")
    print(f" 対象枚数: {len(files)} 枚")
    print("==================================================")

    success_count = 0

    for i, filename in enumerate(files):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        try:
            with Image.open(in_path) as img:
                # iPhone等のExif回転情報を正しく適用（画像の向き崩れ防止）
                img = ImageOps.exif_transpose(img)
                
                orig_w, orig_h = img.size

                # すでに長辺が max_size 以下の場合は縮小不要（比率維持）
                # thumbnail メソッドでアスペクト比を維持しつつ長辺を max_size に収める
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                new_w, new_h = img.size

                # RGBA（透過PNG等）をJPEG保存しようとした際のエラーを防ぐ処理
                if img.mode in ("RGBA", "P") and filename.lower().endswith(('.jpg', '.jpeg')):
                    img = img.convert("RGB")

                # 保存
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    img.save(out_path, "JPEG", quality=quality, optimize=True)
                else:
                    img.save(out_path, optimize=True)

                print(f"[{i+1}/{len(files)}] {filename} : ({orig_w}x{orig_h}) -> ({new_w}x{new_h})")
                success_count += 1

        except Exception as e:
            print(f"[{i+1}/{len(files)}] 【失敗】 {filename}: {e}")

    print("--------------------------------------------------")
    print(f"完了しました！ (成功: {success_count}/{len(files)} 枚)")
    print(f"保存先: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # フォルダパスやサイズはここで自由に変更できます
    INPUT_FOLDER = "data/images_transformed"  # 台形補正後のフォルダ等
    OUTPUT_FOLDER = "data/images_resized"     # Label Studioへ投入するフォルダ等
    MAX_LENGTH = 1024                        # 長辺のpx数

    resize_images(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        max_size=MAX_LENGTH,
        quality=90
    )