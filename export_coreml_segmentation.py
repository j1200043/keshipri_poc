import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import coremltools as ct

# 【セグメンテーションパイプライン用】手書きピクセル抽出モデルをCoreMLに変換する。
# YOLO検出モデルの export_coreml.py とは別モデル・別用途なので混同しないこと。
#
# 入力: RGB画像 512(幅)x320(高さ)、0-255のピクセル値をそのまま渡せばよい
#       (正規化=/255・ImageNet平均/分散の減算はモデル側に組み込み済み)
# 出力: "mask" という名前の 1x1x320x512 Float配列(シグモイド後の確率 0.0-1.0)
#       Swift側で閾値(目安0.5)を適用して二値マスクにする

CKPT_PATH = "models/segmentation/best.pt"
OUTPUT_PATH = "models/segmentation/HandwritingSegmentation.mlpackage"

IMG_W, IMG_H = 512, 320
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class WrappedModel(nn.Module):
    """前処理(0-255 -> 正規化)と後処理(sigmoid)をグラフに焼き込んだラッパー"""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        # x: 0-255 の RGB画像 (CoreMLのImageType入力からそのまま渡ってくる)
        x = x / 255.0
        x = (x - self.mean) / self.std
        logits = self.unet(x)
        return torch.sigmoid(logits)


def main():
    unet = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=1)
    state_dict = torch.load(CKPT_PATH, map_location="cpu")
    unet.load_state_dict(state_dict)
    unet.eval()

    model = WrappedModel(unet)
    model.eval()

    example_input = torch.rand(1, 3, IMG_H, IMG_W) * 255.0
    traced = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, IMG_H, IMG_W), color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="mask")],
        minimum_deployment_target=ct.target.iOS15,
        convert_to="mlprogram",
    )

    mlmodel.short_description = "手書きピクセル抽出(セグメンテーション)モデル。出力は0-1の確率マップ。"
    mlmodel.input_description["image"] = "512x320のRGB画像(letterbox済み、0-255)"
    mlmodel.output_description["mask"] = "1x1x320x512 の手書き確率マップ(0.5以上を手書きとみなす)"

    mlmodel.save(OUTPUT_PATH)
    print(f"CoreML変換成功: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
