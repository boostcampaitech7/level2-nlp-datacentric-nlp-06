import os
import argparse
import logging
import pandas as pd
from cleanlab.rank import get_label_quality_scores
from calculate_prediction_prob import pred_prob_kfold
from noise_score import noise_score_spacy
logging.basicConfig(level="INFO")


def main(data_path, save_path):
    # 데이터 읽어오기
    logging.info("Reading dataset...")
    data = pd.read_csv(f"{data_path}train.csv")
    
    # 노이즈 정도 계산
    logging.info("Calculating noise scores...")
    data['noise_score'] = data['text'].apply(noise_score_spacy)
    
    # label 품질 점수 계산
    logging.info("Calculating label quality scores...")
    pred_probs = pred_prob_kfold(data, k=5)
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
    
    logging.info("Saving results...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_data.to_csv(f"{save_path}train.csv", index=False)
    validation_data.to_csv(f"{save_path}validation.csv", index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../data/', help='path where data csv is stored')
    parser.add_argument('--save_path', type=str, default='../../datasets/v0.0.2/', help='path for saving dataset')
    args = parser.parse_args()
    
    main(args.data_path, args.save_path)