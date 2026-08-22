import os
import glob
import random
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 手書きピクセル抽出セグメンテーションモデルの学習スクリプト。
# data/images_letterboxed/(512x320に統一済みの画像) と
# data/masks/(Label Studioのbrushアノテーションから生成した二値マスク) のペアを使う。

IMAGE_DIR = "data/images_letterboxed"
MASK_DIR = "data/masks"
CHECKPOINT_DIR = "models/segmentation"
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "best.pt")

VAL_RATIO = 0.15
SEED = 42


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def list_pairs():
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    pairs = []
    for img_path in image_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(MASK_DIR, base + ".png")
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs


class HandwritingDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        return image, mask


def build_transforms():
    normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # 白紙が背景なので、回転で生じる余白は白(255)/マスクは0で埋める
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=255, fill_mask=0, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2),
        normalize,
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        normalize,
        ToTensorV2(),
    ])
    return train_tf, val_tf


class DiceBCELoss(nn.Module):
    """クラス不均衡(手書きピクセルは少数派)に対応するため BCE + Dice を合算する"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + smooth) / (union + smooth)).mean()
        return bce_loss + dice_loss


def dice_score(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(SEED)
    pairs = list_pairs()
    print(f"総ペア数: {len(pairs)}")
    if not pairs:
        print(f"【エラー】{IMAGE_DIR} / {MASK_DIR} に画像・マスクのペアが見つかりません。")
        return

    random.shuffle(pairs)
    n_val = max(8, int(len(pairs) * VAL_RATIO))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"train: {len(train_pairs)}  val: {len(val_pairs)}")

    train_tf, val_tf = build_transforms()
    train_ds = HandwritingDataset(train_pairs, train_tf)
    val_ds = HandwritingDataset(val_pairs, val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = get_device()
    print(f"device: {device}")

    model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=1)
    model.to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += dice_score(logits, masks) * images.size(0)
        val_loss /= len(val_ds)
        val_dice /= len(val_ds)

        scheduler.step()

        marker = ""
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            marker = f"  -> ベスト更新、保存 (val_dice={best_dice:.4f})"

        print(f"[{epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}{marker}")

    print(f"学習完了。ベストval_dice={best_dice:.4f}  保存先: {BEST_CKPT_PATH}")


if __name__ == "__main__":
    main()
