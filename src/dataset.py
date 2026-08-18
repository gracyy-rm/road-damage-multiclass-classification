import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


# --- Custom Road Augmentations for Torchvision ---

class RandomGammaTransform:
    def __init__(self, gamma_range=(0.7, 1.3), p=0.4):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            return TF.adjust_gamma(img, gamma=gamma)
        return img


class RandomShadowTransform:
    def __init__(self, shadow_intensity=(0.4, 0.7), p=0.4):
        self.shadow_intensity = shadow_intensity
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        w, h = img.size
        # Generate arbitrary vertices for a shadow polygon across the road
        x1, y1 = random.randint(0, w), 0
        x2, y2 = random.randint(0, w), h
        x3, y3 = random.randint(0, w), h
        x4, y4 = random.randint(0, w), 0

        # Mask creation
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=255)

        img_np = np.array(img).astype(np.float32)
        mask_np = np.array(mask).astype(np.float32) / 255.0

        intensity = random.uniform(*self.shadow_intensity)
        shadow_factor = 1.0 - (mask_np * (1.0 - intensity))
        img_np = img_np * shadow_factor[:, :, None]

        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


# --- Data Transforms ---

def get_data_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        RandomShadowTransform(shadow_intensity=(0.4, 0.7), p=0.35),
        RandomGammaTransform(gamma_range=(0.7, 1.3), p=0.35),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])

    return train_transform, val_transform


# --- Datasets ---

class RoadDamageDataset(Dataset):
    """
    Custom PyTorch dataset for road damage multiclass classification.
    """
    def __init__(self, df, transform=None):
        self.image_paths = df["image_path"].values
        self.labels = df["class_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class InferenceDataset(Dataset):
    """
    Dataset for batch inference with fallback handling.
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        img_name = os.path.basename(img_path)
        true_label = self.df.iloc[idx]["class_id"] if "class_id" in self.df.columns else -1
        true_class = self.df.iloc[idx]["class_name"] if "class_name" in self.df.columns else "Unknown"

        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image) if self.transform else image
            is_valid = True
        except Exception:
            image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            is_valid = False

        return image_tensor, img_path, img_name, true_label, true_class, is_valid


if __name__ == "__main__":
    print("--- Running Dataset Structure Validation ---")
    train_transform, val_transform = get_data_transforms()
    print("Transforms initialized successfully.")
    print("Dataset classes initialized successfully.")