import os
import sys
import tqdm
import logging
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores

logging.basicConfig(level="INFO")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from noise_score import noise_score_spacy
from llama import Llama


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Clean():        
    def clean_characters(self, data, system_prompt_path, fewshot_path, save_path):
        data['noise_score'] = data['text'].apply(noise_score_spacy)
        
        # 노이즈 점수별 데이터 분류
        data_clean = data[data.noise_score < 30].reset_index(drop=True)
        data_denoise = data[(data.noise_score >= 30) & (data.noise_score < 70)].reset_index(drop=True)
        
        # 노이즈 데이터 재작성 (30 <= noise_score < 70)
        llama = Llama()
        data_denoise['denoised_text'] = ""
        for idx, row in data_denoise.iterrows():
            denoised_text = llama.inference(system_prompt_path, row['text'], fewshot_path)
            data_denoise.loc[idx,'denoised_text'] = denoised_text

        # 노이즈 제거된 데이터 저장
        data_final = data_denoise[data_denoise.denoised_text != "복구 불가"]  # 복구 불가한 텍스트 제외
        data_final['text'] = data_final['denoised_text']  # 재작성된 텍스트로 text 대체
        data_final = data_final[['ID', 'text', 'target']]
        data_final = pd.concat([data_clean, data_final], axis=0)  # denoise 완료 데이터
        
        # 폴더가 없으면 생성
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data_final.to_csv(save_path+"train.csv", index=False)
    
    def clean_labels_cleanlab(self, model_path, data, save_path):
        checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint")]
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1])) if checkpoints else None
        checkpoint_path = os.path.join(model_path, latest_checkpoint)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=7).to(DEVICE)
        model.eval()
        
        # prediction probability 계산
        pred_probs = []
        for idx, row in tqdm(data.iterrows()):
            tokenized_input = tokenizer(row['text'], padding='max_length', max_length=50, truncation=False, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**tokenized_input).logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred_probs.extend(probs.cpu().numpy())
        pred_probs = np.array(pred_probs)
        
        # label 이슈 데이터 삭제
        ordered_label_issues = find_label_issues(
            labels=data['target'],
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence',
        )
        cleaned_data = data[~data.index.isin(ordered_label_issues)].reset_index(drop=True)
        
        # 폴더가 없으면 생성
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cleaned_data.to_csv(save_path+"train.csv", index=False)
        
    def clean_labels_llama(self, model_path, data, system_prompt_path, fewshot_path, save_path):
        # Llama로 자연어 label 정보 생성
        llama = Llama()
        data['inferenced_label'] = ""
        for idx, row in data.iterrows():
            inferenced_label = llama.inference(system_prompt_path, row['text'], fewshot_path)
            data.loc[idx,'inferenced_label'] = inferenced_label
        
        # noise 점수 측정
        data['noise_score'] = data['text'].apply(noise_score_spacy)
        
        # 라벨 품질 점수 측정
        checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint")]
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1])) if checkpoints else None
        checkpoint_path = os.path.join(model_path, latest_checkpoint)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=7).to(DEVICE)
        model.eval()
        
        pred_probs = []
        for idx, row in tqdm(data.iterrows()):
            tokenized_input = tokenizer(row['text'], padding='max_length', max_length=50, truncation=False, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**tokenized_input).logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred_probs.extend(probs.cpu().numpy())
        pred_probs = np.array(pred_probs)
        
        label_quality_scores = get_label_quality_scores(
            labels=data['target'].values,
            pred_probs=pred_probs
        )
        data['label_quality'] = label_quality_scores
        
        # re-labeling
        # llama로 추정된 라벨에 대한 임베딩 계산
        model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
        label_embeddings = model.encode(data['inferenced_label'].tolist())
        
        # 임베딩 클러스터링 수행 (14개 클러스터)
        num_clusters = 14
        kmeans = KMeans(n_clusters=num_clusters, random_state=456)
        data['cluster'] = kmeans.fit_predict(label_embeddings)
        
        # noise_score와 label_quality에 가중치를 두어 상위 10개 데이터의 최빈 target 값을 추출하여 대표 라벨 설정
        cluster_to_label_map = {}
        weight_noise = 0.7  # noise_score 가중치
        weight_quality = 0.3  # label_quality 가중치
        
        for cluster in data['cluster'].unique():
            cluster_data = data[data['cluster'] == cluster].copy()
            
            # noise_score와 label_quality의 가중 합을 기준으로 상위 10개 데이터 추출
            cluster_data['weighted_score'] = (
                weight_noise * cluster_data['noise_score'] + weight_quality * cluster_data['label_quality']*100
            )
            top_10_data = cluster_data.nlargest(10, 'weighted_score')
            
            # 상위 10개 데이터의 최빈 target 값 추출
            most_common_label = Counter(top_10_data['target']).most_common(1)[0][0]
            cluster_to_label_map[cluster] = most_common_label
        
        # noise_score가 30 이하인 데이터에 대해 재라벨링 수행
        data['target_relabelled'] = data.apply(
            lambda row: cluster_to_label_map[row['cluster']] if row['noise_score'] < 30 else row['target'],
            axis=1
        )
        
        # Relabelled 데이터 개수 조회
        total_length = len(data)
        relabelled_data = len(data[data.target != data.target_relabelled])
        logging.info(f"재라벨링된 데이터 개수: {relabelled_data}개 / {total_length}개")
        
        # relabelled 데이터 저장
        relabelled_data = data.copy()
        relabelled_data['target'] = relabelled_data['target_relabelled']
        relabelled_data = relabelled_data[['ID', "text", "target"]]
        
        # 폴더가 없으면 생성
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        relabelled_data.to_csv(save_path+"train.csv", index=False)


    def plot_noise_scores(self, data_path, save_path, filename):        
        # noise score 계산
        data = pd.read_csv(data_path)
        data['noise_score'] = data['text'].apply(noise_score_spacy)
        
        plt.figure(figsize=(8, 3))
        
        # noise score 히스토그램 시각화
        plt.subplot(1, 2, 1)
        plt.hist(data['noise_score'], bins=30, color='skyblue', alpha=0.7, label='Original Data')
        plt.axvline(x=30, color='orange', linestyle='--', label='Noise Score Threshold (30)')
        plt.axvline(x=70, color='red', linestyle='--', label='Noise Score Threshold (70)')
        plt.xlim(0,100)
        plt.ylim(0, 450)
        plt.xlabel("Noise Score")
        plt.ylabel("Frequency")
        plt.title("Histogram of Noise Scores (Original Data)")
        plt.legend()
        
        # noise score boxplot 시각화
        plt.subplot(1, 2, 2)
        plt.boxplot(data['noise_score'], vert=False)
        plt.xlim(0,100)
        plt.xlabel("Noise Score")
        plt.title("Box Plot of Noise Scores (Original Data)")
        
        # 폴더가 없으면 생성
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename))
        plt.close()