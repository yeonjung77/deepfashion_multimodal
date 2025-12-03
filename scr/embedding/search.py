import argparse
import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np

import faiss
from PIL import Image
import torch

from scr.data.preprocess import val_transform
from scr.embedding.utils_model import load_feature_extractor


def _load_metadata(meta_path: Path) -> List[Dict]:
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def _load_index(index_path: Path) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    return faiss.read_index(str(index_path))


def _prepare_query_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return val_transform(image).unsqueeze(0)



def search_similar(
    query_image: str,
    top_k: int = 5,
    artifact_dir: str = "outputs/embeddings",
    checkpoint_path: str = "outputs/checkpoints/resnet50_finetune2.pth",
) -> List[Dict]:
    artifact_root = Path(artifact_dir)
    metadata = _load_metadata(artifact_root / "meta.pkl")
    index = _load_index(artifact_root / "index.faiss")

    feature_extractor = load_feature_extractor(checkpoint_path)
    query_tensor = _prepare_query_tensor(Path(query_image))
    query_feat = feature_extractor(query_tensor).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(query_feat)
    scores, indices = index.search(query_feat, top_k)
    results: List[Dict] = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        info = metadata[int(idx)]
        results.append(
            {
                "rank": rank,
                "image_name": info["image_name"],
                "image_path": info["image_path"],
                "score": float(score),
                "attributes": info.get("attributes"),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Search similar images using FAISS index")
    parser.add_argument("query", type=str, help="Path to the query image")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--artifact-dir", type=str, default="outputs/embeddings")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/resnet50_finetune2.pth")
    args = parser.parse_args()

    results = search_similar(
        query_image=args.query,
        top_k=args.top_k,
        artifact_dir=args.artifact_dir,
        checkpoint_path=args.checkpoint,
    )

    print("Top-k results:")
    for item in results:
        print(f"[{item['rank']}] {item['image_name']} (score={item['score']:.4f}) -> {item['image_path']}")


if __name__ == "__main__":
    main()
