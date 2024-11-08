
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split

from utils import *

# %% [markdown]
# ## Set Hyperparameters

# %%
SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# %%
## for wandb setting
os.environ['WANDB_DISABLED'] = 'true'


f1 = evaluate.load('f1')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')

class RLM():
    def __init__(self, config):
        self.config = config

        self.output_dir = config["rlm"]["model_output_dir"]
        self.model_name = config["rlm"]["model_name"]
        print("rlm model_name", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=7)
        self.device = config["device"]

        self.batch_size = config["rlm"]["batch_size"]

        data_list = []
        file_name_list = []
        for file in sorted(list(Path(config['preprocess_data_folder']).glob("*_train.csv")), key=lambda f:f.name):
            if "rlm" in file.name or config["ascii_ratio"]+"_" not in file.name:continue
            data = pd.read_csv(file)
            data_list += [data]
            file_name_list += [file.name]
        print("num of train data files:", len(data_list))
        print("train data files:", file_name_list)
        train_data = pd.concat(data_list)

        data_list = []
        file_name_list = []
        for file in sorted(list(Path(config['preprocess_data_folder']).glob("*_valid.csv")), key=lambda f:f.name):
            if "rlm" in file.name or config["ascii_ratio"]+"_" not in file.name:continue
            data = pd.read_csv(file)
            data_list += [data]
            file_name_list += [file.name]
        print("num of valid data files:", len(data_list))
        print("valid data files:", file_name_list)
        valid_data = pd.concat(data_list)

        test_file_name = f"{config['preprocess_data_folder']}/{config['pivot_name']}_lower_{config['ascii_ratio']}.csv"
        test_data = pd.read_csv(test_file_name)
        
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        tokenizer = self.tokenizer
        model = self.model
        
        train_data = self.train_data
        valid_data = self.valid_data

        train_dataset = BERTDataset(train_data, tokenizer)
        valid_dataset = BERTDataset(valid_data, tokenizer)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=True,
            logging_strategy='steps',
            eval_strategy='steps',
            save_strategy='steps',
            logging_steps=100,
            eval_steps=100,
            save_steps=100,
            save_total_limit=2,
            learning_rate= 2e-05,
            adam_beta1 = 0.9,
            adam_beta2 = 0.999,
            adam_epsilon=1e-08,
            weight_decay=0.01,
            lr_scheduler_type='linear',
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            greater_is_better=True,
            seed=SEED
        )
        
        self.trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    def train(self):
        self.trainer.train()
        self.validate()
        self.trainer.save_model(self.output_dir)

    def validate(self):
        print(self.trainer.evaluate())

    def test(self, data=None):
        config = self.config

        tokenizer = self.tokenizer
        model = self.model
        
        model.eval()
        model.to(self.device)

        dataset_test = self.test_data if data is None else data

        preds = []
        for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
            inputs = tokenizer(sample['text'], return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
                preds.extend(pred)

        dataset_test['target'] = preds
        dataset_test.sort_values(by='ID').to_csv(f"{config['preprocess_data_folder']}/{config['pivot_name']}_lower_{config['ascii_ratio']}_rlm.csv", index=False)



        