import os
import sys
import glob
import argparse

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 学習済みモデル(models/segmentation/best.pt)で手書きピクセルを抽出し、
# 1) 検出マスクの重ね合わせプレビュー
# 2) 手書きピクセルを白で消したクリーン画像
# を出力する。data/images_letterboxed/ と同じ512x320画像を入力として想定。

CKPT_PATH = "models/segmentation/best.pt"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=1)
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess(image_bgr):
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return transform(image=image_rgb)["image"]


def predict_mask(model, device, image_bgr, threshold=0.5):
    tensor = preprocess(image_bgr).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (probs > threshold).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", help="入力画像 (data/images_letterboxed/ と同サイズ想定)")
    parser.add_argument("--out-dir", default="data/predict_output")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = get_device()
    model = load_model(device)

    os.makedirs(args.out_dir, exist_ok=True)

    files = []
    for pattern in args.input:
        files.extend(sorted(glob.glob(pattern)))

    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"【スキップ】読み込めません: {path}")
            continue

        mask = predict_mask(model, device, img, args.threshold)
        base = os.path.splitext(os.path.basename(path))[0]

        overlay = img.copy()
        overlay[mask > 0] = (0, 0, 255)
        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        cv2.imwrite(os.path.join(args.out_dir, base + "_overlay.jpg"), blended)

        cleaned = img.copy()
        cleaned[mask > 0] = (255, 255, 255)
        cv2.imwrite(os.path.join(args.out_dir, base + "_cleaned.jpg"), cleaned)

        print(f"[OK] {path} -> {base}_overlay.jpg / {base}_cleaned.jpg")


if __name__ == "__main__":
    main()
