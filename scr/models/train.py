import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from github.deepfashion_multimodal.src.models.metrics import compute_metrics

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_probs, all_labels = [], []

    pbar = tqdm(
        train_loader,
        desc="[Train]",
        dynamic_ncols=True,
        ascii=True,
        position=0,
    )

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labs = labels.cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labs)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    acc, macro_f1, _ = compute_metrics(all_labels, all_probs)
    return total_loss / len(train_loader), acc, macro_f1


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(
            val_loader,
            desc="[Val]",
            dynamic_ncols=True,
            ascii=True,
            position=0,
        )
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            labs  = labels.cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labs)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    acc, macro_f1, _ = compute_metrics(all_labels, all_probs)
    return total_loss / len(val_loader), acc, macro_f1


def training_loop(model, train_loader, val_loader, optimizer, scheduler, 
                  device, criterion, epochs, save_path):
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": []
    }

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch} =====")
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Model saved:", save_path)

    return history
