import argparse
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scr.data.dataset import DeepFashionAttrDataset
from scr.data.preprocess import val_transform
from scr.embedding.utils_model import load_feature_extractor


class DatasetWithMeta(Dataset):
    """Wrap DeepFashionAttrDataset to expose metadata."""

    def __init__(self, base_dataset: DeepFashionAttrDataset, image_root: str):
        self.base = base_dataset
        self.df = base_dataset.df
        self.image_root = image_root

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, labels = self.base[idx]
        row = self.df.iloc[idx]
        image_name = row["image_name"]
        meta = {
            "image_name": image_name,
            "image_path": os.path.join(self.image_root, image_name),
            "row_idx": idx,
        }
        return img, labels, meta


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_embeddings(
    csv_path: str,
    image_root: str,
    checkpoint_path: str,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
) -> Tuple[np.ndarray, List[dict]]:
    base_ds = DeepFashionAttrDataset(csv_path, image_root, val_transform)
    dataset = DatasetWithMeta(base_ds, image_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    feature_extractor = load_feature_extractor(checkpoint_path)

    all_features: List[np.ndarray] = []
    metadata: List[dict] = []

    with torch.no_grad():
        for imgs, labels, metas in tqdm(loader, desc="Extracting embeddings"):
            feats = feature_extractor(imgs)
            all_features.append(feats.cpu().numpy())
            labels_np = labels.numpy()

            for i in range(feats.size(0)):
                entry = {
                    "image_name": metas["image_name"][i],
                    "image_path": metas["image_path"][i],
                    "row_idx": int(metas["row_idx"][i]),
                    "attributes": labels_np[i].tolist(),
                }
                metadata.append(entry)

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    _ensure_dir(output_dir)
    np.save(output_dir / "features.npy", features)
    with open(output_dir / "meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return features, metadata

def build_faiss_index(features: np.ndarray, index_path: Path) -> faiss.Index:
    faiss.normalize_L2(features)
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    faiss.write_index(index, str(index_path))
    return index


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings & build a FAISS index")
    parser.add_argument("--csv", type=str, default="data/processed/val.csv")
    parser.add_argument("--image-root", type=str, default="data/raw/images")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/resnet50_finetune2.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="outputs/embeddings")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    features, _ = extract_embeddings(
        csv_path=args.csv,
        image_root=args.image_root,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=output_dir,
    )

    index_path = output_dir / "index.faiss"
    build_faiss_index(features, index_path)
    print(f"Saved features/meta under {output_dir}")
    print(f"Saved index to {index_path}")


if __name__ == "__main__":
    main()
