# DeepFashion-MultiModal Attribute Tagging & Style Retrieval
##### Transfer Learning + Embedding-based Fashion Style Search System

DeepFashion-MultiModal 데이터를 활용해 패션 이미지 Attribute 태깅(Shape/Fabric/Color) 모델을 구축했습니다. Fine-Tuning 기반 딥러닝 모델로 18차원 속성 벡터를 예측하고, 임베딩 + FAISS를 활용해 유사 스타일 검색 시스템(Streamlit 데모)을 구현했습니다.

##### 주요기능
- `Multi-label Attribute Tagging`: ResNet50 기반 Fine-Tuning으로 18차원 패션 속성(shape, fabric, color) 자동 태깅
- `Embedding Extraction`: Feature layer에서 2048-D 이미지 임베딩 벡터 추출
- `FAISS Similarity Search`: Query 이미지와 가장 유사한 패션 아이템 Top-K 검색
- `Streamlit Web Demo`: 이미지 업로드 → 태깅 결과 → 유사 아이템 확인 가능한 웹 서비스

##### Models & RAG
- Base Backbone : ResNet50 (ImageNet pretrained)
- Transfer Learning Strategy : Fine-Tuning
- Output : multi-label classification

##### Pipeline
1. Label Parsing — shape/fabric/color txt를 병합해 18차원 Attribute 벡터 생성
2. Preprocessing — Resize·Normalize·Augmentation 적용
3. Attribute Tagging Model — ResNet50 Fine-Tuning으로 18개 속성 예측
4. Embedding Extraction — 모델 feature layer에서 2048-D 이미지 임베딩 추출
5. FAISS Retrieval — Query 이미지 기반 Top-K 스타일 유사 아이템 검색

##### Demo Image