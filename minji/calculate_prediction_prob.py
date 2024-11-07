import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
import logging
logging.basicConfig(level="INFO")

import torch
from torch.utils.data import Dataset, DataLoader
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer


SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_name = 'klue/bert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
os.environ['WANDB_DISABLED'] = 'true'

class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', max_length=50, truncation=False, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }

    def __len__(self):
        return len(self.labels)

f1 = evaluate.load('f1')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')

def pred_prob_kfold(data, k=5):
    # K-Fold Cross Validation 설정
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)

    # validation set의 prediction probability 저장 리스트
    all_pred_probs = np.zeros((len(data), 7))

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        logging.info(f"Training fold {fold + 1}/{k}...")
        
        # Train/Validation split
        train_data = data.iloc[train_idx].reset_index(drop=True)
        val_data = data.iloc[val_idx].reset_index(drop=True)
        
        # Dataset 준비 (main 구성과 동일)
        train_dataset = BERTDataset(train_data, tokenizer)
        val_dataset = BERTDataset(val_data, tokenizer)
        
        # Trainer 설정 (main 구성과 동일)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
        training_args = TrainingArguments(
            output_dir=f'./results_{fold}',
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=True,
            logging_strategy='epoch',
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_steps=100,
            save_total_limit=1,
            learning_rate= 2e-05, # 가능
            adam_beta1 = 0.9, # 불가
            adam_beta2 = 0.999, # 불가
            adam_epsilon=1e-08, # 불가
            weight_decay=0.01, # 불가
            lr_scheduler_type='linear', # 불가
            per_device_train_batch_size=32, # 가능
            per_device_eval_batch_size=32, # 32인 건 이유가 있다.
            num_train_epochs=2, # 불가
            metric_for_best_model='eval_f1',
            greater_is_better=True,
            seed=SEED,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )
        
        # 모델 학습
        trainer.train()
        
        # Validation 데이터에 대해 예측 확률 계산
        val_loader = DataLoader(val_dataset, batch_size=16)
        model.eval()
        pred_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Evaluating fold {fold + 1}"):
                inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred_probs.extend(probs.cpu().numpy())
        
        # 예측 확률 저장
        all_pred_probs[val_idx] = np.array(pred_probs)
        
    return all_pred_probs