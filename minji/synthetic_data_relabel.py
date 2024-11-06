import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import logging
from pprint import pprint
from cleanlab.dataset import health_summary
from cleanlab.filter import find_label_issues

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_path = "../../datasets/v1.3.2/"
save_path = "../../datasets/v1.3.3/"

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


train_data = pd.read_csv(f"{data_path}train.csv")
val_data = pd.read_csv(f"{data_path}validation.csv")

# Dataset 준비
train_dataset = BERTDataset(train_data, tokenizer)
val_dataset = BERTDataset(val_data, tokenizer)

# Trainer 설정
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
training_args = TrainingArguments(
    output_dir=f'./outputs',
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
    seed=SEED, # 불가? 가급적 건드리지 말기
)

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 모델 학습
trainer.train()

# 학습 결과 평가
eval_results = trainer.evaluate(eval_dataset=val_dataset)
logging.info(f"F1 Score: {eval_results['eval_f1']:.4f}")



model.eval()

# Train 데이터에 대해 예측 확률 계산
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=data_collator)
train_pred_probs = []

with torch.no_grad():
    for batch in tqdm(train_loader):
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'labels'}
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        train_pred_probs.extend(probs.cpu().numpy())

train_pred_probs = np.array(train_pred_probs)

# 전체적인 라벨 품질 측정
class_names=[0,1,2,3,4,5,6]
health_res = health_summary(train_data['target'], train_pred_probs, class_names=class_names)
logging.info(f"Overall label health score: {health_res['overall_label_health_score']}")
pprint(health_res['classes_by_label_quality'])

ordered_label_issues = find_label_issues(
    labels=train_data['target'].values,
    pred_probs=train_pred_probs,
    return_indices_ranked_by='self_confidence',
)
issue_indices = set(ordered_label_issues)
clean_train = train_data[~train_data.index.isin(issue_indices)]

clean_train.to_csv(f"{save_path}train.csv", index=False)
logging.info(f"Deleted texts: {len(issue_indices)}")