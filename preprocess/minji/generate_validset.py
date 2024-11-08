import os
import sys
from tqdm import tqdm
import argparse
import logging
import pandas as pd
import numpy as np
from cleanlab.rank import get_label_quality_scores
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
logging.basicConfig(level="INFO")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from src.noise_score import noise_score_spacy

        
def main(data_path, model_path, save_path):
    # 데이터 읽어오기
    logging.info("Reading dataset...")
    data = pd.read_csv(f"{data_path}train.csv")
    
    # 노이즈 정도 계산
    logging.info("Calculating noise scores...")
    data['noise_score'] = data['text'].apply(noise_score_spacy)
    
    # label 품질 점수 계산
    logging.info("Calculating label quality scores...")
    checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint")]
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1])) if checkpoints else None
    checkpoint_path = os.path.join(model_path, latest_checkpoint)
    
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=7).to(DEVICE)
    model.eval()
    
    # prediction probability 계산
    pred_probs = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
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
    
    # validation dataset 추출
    logging.info("Extracting validation set...")
    data_clear = data[data.noise_score < 30] # 노이즈 거의 없는 데이터
    validation_data = pd.concat([  # 각 target별로 상위 label_quality 데이터를 추출
        data_clear[data_clear['target'] == target].nlargest(20, 'label_quality') for target in range(7)
    ], ignore_index=True)
    
    # 나머지는 train data로 저장
    train_data = data[~data.ID.isin(validation_data.ID)]
    
    # 불필요한 열 삭제
    cols = ["ID", "text", "target"]
    train_data = train_data[cols]
    validation_data = validation_data[cols]
    
    logging.info("Saving results...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_data.to_csv(f"{save_path}train.csv", index=False)
    validation_data.to_csv(f"{save_path}validation.csv", index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../data/', help='path where data csv is stored')
    parser.add_argument('--model_path', type=str, default='../model')
    parser.add_argument('--save_path', type=str, default='../../datasets/v0.0.2/', help='path for saving dataset')
    args = parser.parse_args()
    
    main(args.data_path, args.model_path, args.save_path)