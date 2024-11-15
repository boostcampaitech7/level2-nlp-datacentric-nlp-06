import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.notebook import tqdm
from cleanlab.filter import find_label_issues
import glob
import argparse

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_datasets(directory_path):
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def apply_cleanlab(train, model_path, model_name, display_issues=False):
    # cleanlab 사용해서 relabeling 하는 함수
    # model_path: cleanlab에 사용되는 model 경로 매개변수
    # display_issues: relabeling 값 출력 여부 정하는 매개변수
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path + model_name, num_labels=7).to(DEVICE)
    
    model.eval()
    preds = []
    for _, data in tqdm(train.iterrows(), total=len(train), desc="Generating predictions"):
        tokenized = tokenizer(data['text'], padding='max_length', max_length=50, truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**tokenized).logits.cpu().numpy()
            softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            preds.append(softmax[0])

    pred_probs = np.array(preds)
    label_issues = find_label_issues(labels=train['target'], pred_probs=pred_probs, return_indices_ranked_by='self_confidence')

    if display_issues:
        print(f"Total issues detected: {len(label_issues)}")
        for issue in label_issues:
            print(f"Text: {train.iloc[issue]['text']}")
            print(f"Original Label: {train.iloc[issue]['target']}, Predicted: {np.argmax(pred_probs[issue])}")
            print("-" * 40)
    
    train.loc[label_issues, 'target'] = np.argmax(pred_probs[label_issues], axis=1)
    return train

def remove_duplicates(train, option=0, display_samples=False):
    # 겹치는 컬럼의 데이터 값 합치는 함수
    # display_samples: 겹치는 컬럼의 데이터값 출력 여부 정하는 매개변수
    options = {
        0: ['ID', 'text', 'target'], # ID, text, target 컬럼이 겹치는 경우 합치기
        1: ['text', 'target'], # text, target 컬럼이 겹치는 경우 합치기
        2: ['ID', 'text'] # ID, text 컬러밍 겹치는 경우 합치기
    }

    if option in options:
        train = train.drop_duplicates(subset=options[option], keep='first').reset_index(drop=True)
    else:
        for op in options:
            train = train.drop_duplicates(subset=options[op], keep='first').reset_index(drop=True)

    if display_samples:
        print_duplicate_samples(train)
    return train

def print_duplicate_samples(train, sample_size=50):
    # 겹치는 컬럼의 데이터 값 출력하는 함수
    # sample_size: 몇개 보여줄지 정하는 매개변수
    duplicates = train[train["text"].duplicated(keep=False)]
    duplicate_groups = duplicates.groupby("text")
    print(f"Total duplicate groups: {len(duplicate_groups)}")

    for sentence, group in duplicate_groups.head(sample_size):
        print(f"Sentence: {sentence}")
        print(group)
        print("-" * 50)

def main(directory_path, model_path, model_name):
    train = load_datasets(directory_path)
    train = apply_cleanlab(train, model_path, model_name, display_issues=True)
    train = remove_duplicates(train)
    train.to_csv(directory_path + "/cleaned_train_data.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../dataset', help='path where data csv is stored')
    parser.add_argument('--model', type=str, default='../model', help='path for saving model during training and loading during testing')
    parser.add_argument('--model_name', type=str, default="/checkpoint-518", help="model name in your model_path")
    args = parser.parse_args()
    
    main(args.data, args.model, args.model_name)