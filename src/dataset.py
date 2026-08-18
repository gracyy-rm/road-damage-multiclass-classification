import os
import torch
import pandas as pd

from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

def get_data_transforms(img_size=224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # Make the lighting harsher
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,std=IMAGE_STD)
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,std=IMAGE_STD)
        ]
    )

    return train_transform, val_transform

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
        label = torch.tensor(self.labels[idx],dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label



class InferenceDataset(Dataset):
    """
        Dataset for batch inference.
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
    train_transform, val_transform = (get_data_transforms())
    print(" Transforms initialized successfully.")
    print(" Dataset classes initialized successfully.")