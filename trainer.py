import os
import torch
import random
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import BERTDataset

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MyTrainer:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

        self.f1 = evaluate.load('f1')

    def train(self):
        model_name = 'klue/bert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

        data = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        train, valid_ = train_test_split(data, test_size=0.3, random_state=SEED) # train, test 비율 수정 가능
        train = BERTDataset(train, tokenizer)
        valid = BERTDataset(valid_, tokenizer)

        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=True,
            logging_strategy='steps',
            eval_strategy='steps',
            save_strategy='epoch',
            logging_steps=100,
            eval_steps=100,
            # save_steps=100,
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
            # load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            greater_is_better=True,
            seed=SEED, # 불가? 가급적 건드리지 말기
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=valid,
            data_collator=collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        tokenizer.save_pretrained(os.path.join(self.model_path))

        self.test(data=valid_)

    def test(self, data=None):
        if data is None:
            test = pd.read_csv(os.path.join(self.data_path, 'test.csv'))
        else:
            test = data

        if not os.path.isdir(self.model_path):
            print(f"No model in {self.model_path}")
            return

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(self.model_path, 'checkpoint-124'), num_labels=7).to(DEVICE)
        
        model.eval()
        preds = []

        for idx, sample in tqdm(test.iterrows(), total=len(test), desc='Evaluating'):
            inputs = tokenizer(sample['text'], return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
                preds.extend(pred)
        
        test['target'] = preds
        test.to_csv(os.path.join(self.data_path, 'output.csv'), index=False)
        print(f"Saved predictions to {os.path.join(self.data_path, 'output.csv')}")
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.f1.compute(predictions=predictions, references=labels, average='macro')
