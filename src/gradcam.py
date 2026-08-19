import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .model import create_model
from .dataset import get_data_transforms


class RoadDamageGradCAM:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.model_name = self.checkpoint["architecture"]
        self.config = self.checkpoint["config"]
        self.image_size = self.config["training_parameters"]["image_size"]
        self.class_names = {int(k): v for k, v in self.config["classes"]["mapping"].items()}
        self.model = self._load_model()
        _, self.transform = get_data_transforms(self.image_size)

    def _load_model(self):
        model = create_model(
            model_name=self.model_name,
            num_classes=len(self.class_names),
            freeze_backbone=False
        )
        model.load_state_dict(self.checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def get_target_layer(model, model_name):
        if model_name == "resnet18":
            return model.backbone.layer4[-1]
        elif model_name in ("mobilenet_v3_small", "efficientnet_b0"):
            return model.backbone.features[-1]
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def inspect_single_df(df, model, model_name, device, num_samples=3):
        if df.empty:
            print("Skipping: DataFrame is empty.")
            return

        print("=" * 55)
        print(f"GRAD-CAM INSPECTION ({len(df):,} images)")
        print("=" * 55)

        model = model.to(device)
        model.eval()
        _, transform = get_data_transforms(img_size=224)
        target_layer = RoadDamageGradCAM.get_target_layer(model, model_name)
        cam = GradCAM(model=model, target_layers=[target_layer])

        sampled_df = df.sample(n=min(num_samples, len(df)), random_state=42)

        for _, row in sampled_df.iterrows():
            image_path = row["image_path"]
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            image_resized = image.resize((224, 224))
            image_np = np.array(image_resized)

            image_tensor = transform(image).unsqueeze(0).to(device)
            grayscale_cam = cam(input_tensor=image_tensor)[0]
            overlay = show_cam_on_image(image_np.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)

            true_class = row.get("true_class", "Unknown")
            pred_class = row.get("pred_class_name", "Unknown")
            confidence = row.get("confidence_score", 0.0)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(image_np)
            axes[0].set_title(f"True: {true_class}")
            axes[0].axis("off")

            axes[1].imshow(grayscale_cam, cmap="jet")
            axes[1].set_title(f"Pred: {pred_class}\nConf: {confidence:.1f}%")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()