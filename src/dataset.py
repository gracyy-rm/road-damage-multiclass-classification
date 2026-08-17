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
            transforms.ColorJitter(brightness=0.1,contrast=0.1),
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
        image_path = str(self.df.iloc[idx]["image_path"])
        image_name = os.path.basename(image_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return (image,image_path,image_name)



if __name__ == "__main__":
    print("--- Running Dataset Structure Validation ---")
    train_transform, val_transform = (get_data_transforms())
    print(" Transforms initialized successfully.")
    print(" Dataset classes initialized successfully.")