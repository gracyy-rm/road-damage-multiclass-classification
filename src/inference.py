import os
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

from .model import create_model
from .dataset import InferenceDataset,get_data_transforms

class RoadDamageInference:

    def __init__(self,model_path):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self._load_checkpoint_metadata()

        self.transform = self._build_transform()

        self.model = self._load_model()

    def _load_checkpoint_metadata(self):

        if "state_dict" not in self.checkpoint:
            raise KeyError("state_dict not found in checkpoint")

        self.model_name = self.checkpoint["architecture"]

        config = self.checkpoint["config"]

        self.image_size = config["training_parameters"]["image_size"]

        self.class_names = {
            int(k): v
            for k, v in config["classes"]["mapping"].items()
        }

        print("\n--- Checkpoint Information ---")
        print(f"Model: {self.model_name}")
        print(f"Image Size: {self.image_size}")
        print(f"Classes: {len(self.class_names)}")
        print(f"Checkpoint: {self.model_path}")
        print("--------------------------------\n")

    def _build_transform(self):
        _, val_transform = get_data_transforms(img_size=self.image_size)
        return val_transform

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
    
    def _preprocess_image(self,image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        return image,image_tensor

    @torch.no_grad()
    def _predict_tensor(self,image_tensor):
        logits = self.model(image_tensor)
        probabilities = torch.softmax(
            logits,
            dim=1
        )
        pred_label = torch.argmax(probabilities,dim=1).item()
        confidence = probabilities[0][pred_label].item() * 100
        pred_class_name = self.class_names[pred_label]
        return (pred_label,pred_class_name,confidence)

    def predict_image(self,image_path):
        image,image_tensor = self._preprocess_image(image_path)

        pred_label,pred_class_name,confidence = (self._predict_tensor(image_tensor))

        return {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "image": image,
            "pred_label": pred_label,
            "pred_class_name": pred_class_name,
            "confidence_score": round(confidence,2)
        }

    @torch.no_grad()
    def predict_csv_batch(self,csv_path,batch_size=32,num_workers=4):

        input_df = pd.read_csv(csv_path)
        dataset = InferenceDataset(input_df,self.transform )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda"
        )

        results = []

        for batch_tensors,batch_paths,batch_names,batch_true_labels,batch_true_classes,batch_valid in tqdm(dataloader,desc="Inference"):
            batch_tensors = batch_tensors.to(self.device)
            logits = self.model(batch_tensors)
            probabilities = torch.softmax(logits,dim=1)
            predictions = torch.argmax(probabilities,dim=1)

            for i in range(len(batch_paths)):
                if not batch_valid[i]:
                    continue
                pred_label = predictions[i].item()
                confidence = (probabilities[i][pred_label].item()* 100)

                results.append(
                    {
                        "image_path": batch_paths[i],
                        "image_name": batch_names[i],
                        "true_label": batch_true_labels[i].item(),
                        "true_class": batch_true_classes[i],
                        "pred_label": pred_label,
                        "pred_class_name": self.class_names[pred_label],
                        "confidence_score": round(confidence,2)
                    }
                )
        return pd.DataFrame(results)

    