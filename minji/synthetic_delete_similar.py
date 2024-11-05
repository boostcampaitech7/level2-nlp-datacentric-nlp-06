import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

# 데이터 가져오기
data_path = "../../datasets/v1.3.1/"
save_path = "../../datasets/v1.3.2/"
data = pd.read_csv(f"{data_path}train.csv")
train_data = data[data.ID.str.startswith('ynat')]
syn_data = data[~data.ID.str.startswith('ynat')]

# 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 유사도 기준 설정
similarity_threshold = 0.85

# target별 유사도 계산 및 결과 저장
similar_df = {}
for target in range(7):
    # 해당 타겟의 텍스트 필터링
    train_target_data = train_data[train_data['target'] == target]
    syn_target_data = syn_data[syn_data['target'] == target]
    
    # 해당 타겟의 텍스트 임베딩
    train_embeddings = model.encode(train_target_data['text'].tolist(), convert_to_tensor=True)
    syn_embeddings = model.encode(syn_target_data['text'].tolist(), convert_to_tensor=True)

    # 유사도 계산 및 결과 저장용 리스트
    similarity_scores = []
    similar_sentences = {'Train_ID': [], 'Syn_ID': [], 'Train_Text': [], 'Syn_Text': [], 'Cosine_Similarity': []}
    
    # 유사도 계산
    for i, train_emb in enumerate(train_embeddings):
        for j, syn_emb in enumerate(syn_embeddings):
            cos_sim = F.cosine_similarity(train_emb, syn_emb, dim=0).item()
            similarity_scores.append(cos_sim)

            # 유사도가 기준 이상인 경우 저장
            if cos_sim >= similarity_threshold:
                similar_sentences['Train_ID'].append(train_target_data.iloc[i]['ID'])
                similar_sentences['Syn_ID'].append(syn_target_data.iloc[j]['ID'])
                similar_sentences['Train_Text'].append(train_target_data.iloc[i]['text'])
                similar_sentences['Syn_Text'].append(syn_target_data.iloc[j]['text'])
                similar_sentences['Cosine_Similarity'].append(cos_sim)

    # # Target 별 코사인 유사도 분포 시각화
    # plt.figure(figsize=(5, 3))
    # sns.histplot(similarity_scores, bins=20, kde=True)
    # plt.xlabel('Cosine Similarity')
    # plt.ylabel('Frequency')
    # plt.title(f'Cosine Similarity Distribution for Target {target}')
    # plt.show()

    # 유사도가 높은 항목들을 데이터프레임으로 저장
    similar_df[target] = pd.DataFrame(similar_sentences)

# 유사도가 임계값 이상인 데이터 ID 추출
syn_to_delete = []
for i in range(7):
    to_delete = list(similar_df[i].Syn_ID)
    syn_to_delete.extend(to_delete)

# 해당 데이터 삭제 및 저장
data_sim_del = data[~data.ID.isin(syn_to_delete)]
data_sim_del.to_csv(f"{save_path}train.csv", index=False)