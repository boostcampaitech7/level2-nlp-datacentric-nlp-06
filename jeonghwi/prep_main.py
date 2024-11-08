import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import json
import argparse
import torch

from src.noiseCalc import get_noise_df
from src.BERT_masking import MLM_Filter
from src.gemma_Filter import GemmaFilterModel
from src.gemma_augmentation import GemmaAugModel
from src.Backtranslation import BTModel
from src.SBERT_clustering import SBERT

# Seed Set
SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# 디바이스 설정 (GPU가 사용 가능하면 GPU를 사용하고, 그렇지 않으면 CPU 사용)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



if __name__ == "__main__":
    
    # load data
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",default="../data")
    parser.add_argument("--filter",default="gemma")
    args = parser.parse_args()
    
    data = pd.read_csv(os.path.join(args.data_path,"train.csv"))
    
    # Noise ratio 30% under data
    data_ratio_30 = get_noise_df(data, 30, types="down")

    # 1. Filter Select    
    match args.filter:
        # BART Mask Filter option (30% 이하 데이터)
        case "BART":
            filter_model = MLM_Filter(data_ratio_30)
            filtered_data = filter_model.BART_inference()
        # Gemma Filter option (30% 아래 데이터)
        case "gemma":
            filter_model = GemmaFilterModel(data_ratio_30)
            filtered_data = filter_model.inference_filter()
    # 2. SBERT Relabeling
    relabel_model = SBERT(filtered_data)
    kmeans = relabel_model.clustering(k=3)
    relabel_data = relabel_model.mapping(kmeans)

    # 3. Augmentation Select
    # Back Translation
    backtrans_model = BTModel(relabel_data)
    back_aug_data = backtrans_model.backtranslation()
    back_aug_data = back_aug_data.sample(300,random_state=SEED) # 1504개 중 300개만 추출
    
    # merge original data
    back_concat_data = pd.concat([relabel_data, back_aug_data])

    # Gemma Prompting
    gemma_aug_model = GemmaAugModel(relabel_data)
    gemma_aug_data = gemma_aug_model.inference_aug()

    gemma_concat_data = pd.concat([back_concat_data, gemma_aug_data])
    gemma_concat_data.to_csv(os.path.join(args.data_path,"train_prep_aug.csv"),encoding="utf-8-sig")
    print("Preprocessing & Augmentation Complete")