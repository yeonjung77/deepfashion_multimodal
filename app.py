import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from PIL import Image

import faiss
import torch

from scr.data.preprocess import val_transform
from scr.embedding.utils_model import load_feature_extractor


# =========================================
# 1) GLOBAL LOAD (FAISS + metadata + model)
# =========================================

ARTIFACT_DIR = Path("outputs/embeddings")
CHECKPOINT_PATH = "outputs/checkpoints/resnet50_finetune2.pth"
IMAGE_ROOT = Path("data/raw/images")

INDEX_PATH = ARTIFACT_DIR / "index.faiss"
META_PATH = ARTIFACT_DIR / "meta.pkl"

# Load metadata
with open(META_PATH, "rb") as f:
    META = pickle.load(f)

# Load FAISS index
INDEX = faiss.read_index(str(INDEX_PATH))

# Load feature extractor (ResNet50 finetuned)
MODEL = load_feature_extractor(CHECKPOINT_PATH)
MODEL.eval()


# =========================================
# 2) Utility: 임베딩 추출 함수
# =========================================
def extract_feature(image: Image.Image):
    tensor = val_transform(image).unsqueeze(0)
    feat = MODEL(tensor).cpu().detach().numpy().astype(np.float32)
    faiss.normalize_L2(feat)
    return feat


# =========================================
# 3) 검색 함수
# =========================================
def search_similar(image: Image.Image, top_k=5):
    feat = extract_feature(image)
    distances, indices = INDEX.search(feat, top_k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        item = META[int(idx)]

        # meta.pkl 안에 image_path가 없으면 filename 기반으로 경로 생성
        img_path = item.get("image_path", None)
        if img_path is None or not Path(img_path).exists():
            img_path = IMAGE_ROOT / item["image_name"]

        results.append({
            "rank": rank,
            "image_name": item["image_name"],
            "image_path": str(img_path),
            "score": float(dist),
            "attributes": item.get("attributes")
        })

    return results


# =========================================
# 4) Streamlit UI
# =========================================
st.set_page_config(page_title="DeepFashion Image Search", layout="wide")

st.markdown("## 👗 DeepFashion Image Search Demo")

uploaded_file = st.file_uploader("이미지를 업로드하세요.", type=["jpg", "png", "jpeg"])

top_k = st.slider("Top-K 결과 개수", min_value=3, max_value=10, value=5)


if uploaded_file:

    # ======================
    # Query 이미지 표시
    # ======================
    query_img = Image.open(uploaded_file).convert("RGB")

    st.markdown("### Uploaded Image")
    st.image(query_img, width=200)

    # ======================
    # 검색 실행
    # ======================
    # st.markdown("### 🔎 Searching Similar Images...")
    results = search_similar(query_img, top_k=top_k)

    # ======================
    # 결과 보여줌
    # ======================
    st.markdown("### Top-K Similar Images")

    cols = st.columns(top_k)

    for col, res in zip(cols, results):
        col.image(res["image_path"], caption=f"Rank {res['rank']} | Score {res['score']:.4f}")

        # if res["attributes"]:
        #     shape = res["attributes"][:12]
        #     fabric = res["attributes"][12:15]
        #     pattern = res["attributes"][15:18]

        #     col.write(f"**Shape:** {shape}")
        #     col.write(f"**Fabric:** {fabric}")
        #     col.write(f"**Pattern:** {pattern}")
