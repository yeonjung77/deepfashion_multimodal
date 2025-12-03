import random
import numpy as np
from pathlib import Path
import torch

from scr.embedding.search import search_similar
from scr.data.dataset import DeepFashionAttrDataset
from scr.data.preprocess import val_transform


# -------------------------------
# 1) Top@K Matching Accuracy
# -------------------------------
def evaluate_topk_accuracy(dataset, k=5, num_samples=100):
    indices = random.sample(range(len(dataset)), num_samples)
    correct = 0

    for idx in indices:
        img, label = dataset[idx]
        img_path = dataset.df.iloc[idx]["image_name"]
        full_img_path = Path(dataset.image_root) / img_path

        results = search_similar(str(full_img_path), top_k=k)

        retrieved_names = [r["image_name"] for r in results]

        if img_path in retrieved_names:
            correct += 1

    return correct / num_samples



# -------------------------------
# 2) Attribute-wise Retrieval Score (18-class multi-label)
# -------------------------------
def evaluate_attribute_retrieval(dataset, k=5, num_samples=100):
    indices = random.sample(range(len(dataset)), num_samples)

    attr_scores = []  

    for idx in indices:
        img, true_label = dataset[idx]  
        img_path = dataset.df.iloc[idx]["image_name"]
        full_img_path = Path(dataset.image_root) / img_path

        results = search_similar(str(full_img_path), top_k=k)

        retrieved_attrs = []
        for r in results:
            if "attributes" in r:
                retrieved_attrs.append(np.array(r["attributes"], dtype=np.float32))
            else:
                continue

        if len(retrieved_attrs) == 0:
            continue

        retrieved_attrs = np.stack(retrieved_attrs)  # (k, 18)

        mean_pred = retrieved_attrs.mean(axis=0)

        attr_acc = (true_label.numpy() == (mean_pred > 0.5)).astype(float)

        attr_scores.append(attr_acc)

    attr_scores = np.stack(attr_scores)  
    return attr_scores.mean(axis=0)      



# -------------------------------
# 실행부
# -------------------------------
if __name__ == "__main__":
    dataset = DeepFashionAttrDataset(
        csv_path="data/processed/val.csv",
        image_root="data/raw/images",
        transform=val_transform
    )

    print("===== Retrieval Evaluation =====")
    print("Top@5 Accuracy:", evaluate_topk_accuracy(dataset, k=5))

    print("Attribute-wise Retrieval Score (18-dim):")
    print(evaluate_attribute_retrieval(dataset, k=5))
