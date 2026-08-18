import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchinfo import summary

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .model import create_model
from .dataset import RoadDamageDataset,get_data_transforms
#train
def train_one_epoch(model,dataloader,criterion,optimizer,device,epoch,writer,accumulation_steps=2):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels_train = []
    all_preds_train = []

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(dataloader,desc=f"Epoch {epoch} [Train]",leave=False)

    for batch_idx, (images, labels) in enumerate(progress_bar):

        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss = loss / accumulation_steps
        loss.backward()

        if ((batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += (loss.item() * accumulation_steps) * images.size(0)
        predictions = torch.argmax(logits,dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += images.size(0)
        all_labels_train.extend(labels.cpu().numpy())
        all_preds_train.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    train_precision = precision_score(all_labels_train,all_preds_train,average="weighted",zero_division=0)
    train_recall = recall_score(all_labels_train,all_preds_train,average="weighted",zero_division=0)
    train_f1 = f1_score(all_labels_train,all_preds_train,average="weighted",zero_division=0)

    writer.add_scalar("Loss/Train",epoch_loss,epoch)
    writer.add_scalar("Accuracy/Train",epoch_acc * 100,epoch)
    writer.add_scalar("Metrics/Train_Precision",train_precision,epoch)
    writer.add_scalar("Metrics/Train_Recall",train_recall,epoch)
    writer.add_scalar("Metrics/Train_F1_Score",train_f1,epoch)

    return (epoch_loss,epoch_acc,train_precision,train_recall,train_f1)


# plot_cm_to_tensor
def plot_confusion_matrix_to_tensor(y_true,y_pred,class_names):
    cm = confusion_matrix(y_true,y_pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
    disp.plot(ax=ax,colorbar=False)
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    fig.canvas.draw()
    image_rgba = np.asarray(fig.canvas.buffer_rgba())
    image_rgb = image_rgba[:, :, :3]
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
    plt.close(fig)
    return image_tensor

#validate
@torch.no_grad()
def validate(model,dataloader,criterion,device,epoch,writer,class_names):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels_val = []
    all_preds_val = []

    progress_bar = tqdm(dataloader,desc=f"Epoch {epoch} [Val]",leave=False)

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = criterion(logits,labels)

        running_loss += (loss.item() * images.size(0))

        predictions = torch.argmax(logits,dim=1)

        correct_predictions += (predictions == labels).sum().item()

        total_samples += images.size(0)

        all_labels_val.extend(labels.cpu().numpy())

        all_preds_val.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    val_precision = precision_score(all_labels_val,all_preds_val,average="weighted",zero_division=0)
    val_recall = recall_score(all_labels_val,all_preds_val,average="weighted",zero_division=0)
    val_f1 = f1_score(all_labels_val,all_preds_val,average="weighted",zero_division=0)

    writer.add_scalar("Loss/Validation",epoch_loss,epoch)
    writer.add_scalar("Accuracy/Validation",epoch_acc * 100,epoch)
    writer.add_scalar("Metrics/Val_Precision",val_precision,epoch)
    writer.add_scalar("Metrics/Val_Recall",val_recall,epoch)
    writer.add_scalar("Metrics/Val_F1_Score",val_f1,epoch)
    cm_image_tensor = plot_confusion_matrix_to_tensor(all_labels_val,all_preds_val,class_names)
    writer.add_image("Confusion_Matrix/Validation",cm_image_tensor,epoch)
    return (epoch_loss, epoch_acc,val_precision,val_recall,val_f1)

#create_checkpoints()
def create_checkpoint(model,config,epoch,train_loss,val_loss,train_acc,val_acc,train_precision,val_precision,train_recall,val_recall,train_f1,val_f1):
    return {
        "architecture": config["model"]["architecture"],
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "config": config,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_precision": train_precision,
        "val_precision": val_precision,
        "train_recall": train_recall,
        "val_recall": val_recall,
        "train_f1": train_f1,
        "val_f1": val_f1
    }

def run_pipeline(config):
    # paths
    paths = config["paths"]
    model_cfg = config["model"]
    train_cfg = config["training_parameters"]
    hardware_cfg = config["hardware"]
    class_names = list(config["classes"]["mapping"].values())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"{model_cfg['architecture']}"
        f"_lr{train_cfg['learning_rate']}"
        f"_bs{train_cfg['batch_size']}"
        f"_{timestamp}"
    )

    run_dir = os.path.join(
        paths["experiments_dir"],
        run_name
    )

    tensorboard_dir = os.path.join(run_dir,"tensorboard")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"),"w") as file:

        json.dump(config,file,indent=4)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on: {device}")
    # df
    train_df = pd.read_csv(paths["train_csv"])
    val_df = pd.read_csv(paths["val_csv"])
    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    # transforms
    train_transform, val_transform = get_data_transforms(img_size=train_cfg["image_size"])
    # datasets
    train_dataset = RoadDamageDataset(train_df, train_transform)
    val_dataset = RoadDamageDataset(val_df,val_transform)
    # dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=hardware_cfg["num_workers"],
        pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=hardware_cfg["num_workers"],
        pin_memory=device.type == "cuda"
    )
    # model creation
    model = create_model(model_name=model_cfg["architecture"],num_classes=model_cfg["num_classes"],freeze_backbone=model_cfg["freeze_backbone"]).to(device)
    # loss
    criterion = nn.CrossEntropyLoss()
    # optimizers
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    # lr scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=3,
        min_lr=1e-6
    )

    print("\n--- MODEL SUMMARY ---\n")

    summary(
        model,
        input_size=(
            train_cfg["batch_size"],
            3,
            train_cfg["image_size"],
            train_cfg["image_size"]
        )
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_path = None
    best_epoch = 0
    early_stopping_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
        "learning_rate": []
    }

    for epoch in range(1,train_cfg["epochs"] + 1):

        previous_lr = optimizer.param_groups[0]["lr"]

        train_results = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            accumulation_steps=train_cfg["accumulation_steps"]
        )

        val_results = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            writer=writer,
            class_names=class_names
        )

        (train_loss,train_acc,train_precision,train_recall,train_f1) = train_results

        (val_loss,val_acc,val_precision,val_recall,val_f1) = val_results

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar(
            "Learning_Rate",
            current_lr,
            epoch
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_precision"].append(train_precision)
        history["val_precision"].append(val_precision)
        history["train_recall"].append(train_recall)
        history["val_recall"].append(val_recall)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["learning_rate"].append(current_lr)

        print(
        f"Epoch [{epoch}/{train_cfg['epochs']}] | "
        f"LR: {current_lr:.1e} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc*100:.2f}% | "
        f"Train Precision: {train_precision:.3f} | "
        f"Train Recall: {train_recall:.3f} | "
        f"Train F1: {train_f1:.3f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc*100:.2f}% | "
        f"Val Precision: {val_precision:.3f} | "
        f"Val Recall: {val_recall:.3f} | "
        f"Val F1: {val_f1:.3f}"
    )
        if current_lr < previous_lr:
            print(
                f"[LR DROP] "
                f"{previous_lr:.2e} → {current_lr:.2e}"
            )

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            early_stopping_counter = 0

            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            best_model_path = os.path.join(
                run_dir,
                f"best_model_acc_{val_acc*100:.1f}.pth"
            )

            checkpoint = create_checkpoint(
                model,
                config,
                epoch,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                train_precision,
                val_precision,
                train_recall,
                val_recall,
                train_f1,
                val_f1
            )

            torch.save(checkpoint,best_model_path)

        else:
            early_stopping_counter += 1
            print(
                f"[PATIENCE] "
                f"{early_stopping_counter}/"
                f"{train_cfg['early_stopping_patience']}"
            )

            if early_stopping_counter >= train_cfg["early_stopping_patience"]:
                print("\nEarly stopping triggered.")
                break

    writer.close()

    with open(os.path.join(run_dir, "history.json"),"w") as file:
        json.dump(history,file,indent=4)

    summary_data = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_model_path": best_model_path,
        "tensorboard_dir": tensorboard_dir
    }

    with open(os.path.join(run_dir, "summary.json"),"w") as file:
        json.dump(summary_data,file,indent=4)
        
    return {
        "run_dir": run_dir,
        "history": history,
        "best_model_path": best_model_path
    }