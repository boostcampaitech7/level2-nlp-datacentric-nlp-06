# %%
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

from cleanlab.filter import find_label_issues
from cleanlab.classification import CleanLearning
from cleanlab.dataset import health_summary

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


class CLMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, config, model, tokenizer, optimizer, criterion):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = 2
        self.batch_size = config['clm']['batch_size']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def fit(self, X, Y):
        self.model.train()
        self.model.to(self.device)

        for epoch in range(self.epochs):
            for x in X:
                x['input_ids'] = x['input_ids'].to(self.device).squeeze(1)
                x['attention_mask'] = x['attention_mask'].to(self.device).squeeze(1)
                x['labels'] = x['labels'].to(self.device).squeeze(1)

                outputs = self.model(**x)
                loss = outputs.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def predict_proba(self, X):
        self.model.eval()
        self.model.to(self.device)

        probs_list = []
        with torch.no_grad():
            for x in X:
                x['input_ids'] = x['input_ids'].to(self.device).squeeze(1)
                x['attention_mask'] = x['attention_mask'].to(self.device).squeeze(1)
                x['labels'] = x['labels'].to(self.device).squeeze(1)
                outputs = self.model(**x)
                probs = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                probs_list += [probs]
        probs = np.concatenate(probs_list, axis=0)
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=-1)

class CLM():
    def __init__(self, config):
        self.config=config
        self.device=config['device']

        self.model_name = config['clm']['model_name']
        print("clm model_name", self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=7).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.train_file_name = f"{config['preprocess_data_folder']+config['pivot_name']}_higher_{config['ascii_ratio']}_train.csv"
        self.valid_file_name = self.train_file_name
        self.test_file_name = self.train_file_name
        self.train_data = pd.read_csv(self.train_file_name)
        self.valid_data = pd.read_csv(self.valid_file_name)
        self.test_data = pd.read_csv(self.test_file_name)

        self.cleanlab_relabel = None

    def predict_probs(self, dataset_train):
        model = self.model

        train_pred_probs=[]

        model.to(self.device)
        dataset_train.to(self.device)

        with torch.no_grad():
            for batch in dataset_train:
                outputs = model(**batch)
                train_pred_probs += [torch.nn.functional.softmax(outputs.logits, dim=-1)]

        train_pred_probs = torch.cat(train_pred_probs, dim=0).detach().cpu().numpy()

        return train_pred_probs
    
    def train(self):
        model = self.model
        tokenizer = self.tokenizer

        train_data = self.train_data
        dataset_train = MyBERTDataset(train_data, tokenizer)

        train_pred_probs = self.predict_probs(dataset_train)

        print("relabeling start")
        model = CLMClassifier(self.config, model, tokenizer, torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01), torch.nn.CrossEntropyLoss())
        cleanlab_relabel = CleanLearning(clf=model, seed=SEED)  # You can pass your PyTorch model
        cleanlab_relabel.fit(MyBERTDataset(train_data, tokenizer), train_data['target'], pred_probs=train_pred_probs)  # You can pass the dataset and predicted probabilities
    
        self.cleanlab_relabel = cleanlab_relabel

    def validate(self):
        model = self.model
        tokenizer = self.tokenizer

        model.eval()
        model.to(self.device)
        
        valid_data = self.valid_data
        dataset_train = MyBERTDataset(valid_data, tokenizer)

        train_pred_probs = self.predict_probs(dataset_train)

        ordered_label_issues = find_label_issues(
            labels=valid_data['target'],
            pred_probs=train_pred_probs,
            return_indices_ranked_by='self_confidence',
        )

        head_issues = ordered_label_issues[:3]
        for issue in head_issues:
            print('input_text:', valid_data.iloc[issue]['text'])
            print('label_text:', valid_data.iloc[issue]['target'])
            print('-------------------------------------------------')
        print("label_issues:",ordered_label_issues)

        class_names = [*range(7)]
        health_summary(valid_data['target'], train_pred_probs, class_names=class_names)

    def test(self):
        #self.relabel_and_discard()
        self.relabel()

    def relabel_and_discard(self):
        tokenizer = self.tokenizer
        model = self.model

        model.eval()
        model.to(self.device)

        cleanlab_relabel = self.cleanlab_relabel

        train_data = self.train_data
    
        new_labels = cleanlab_relabel.predict(MyBERTDataset(train_data, tokenizer))  # Get the new labels for relabeling
        print("num of new_labels:", len(new_labels))
        print("new_labels", new_labels)

        c = sum(1 for a ,b in zip(new_labels, train_data.target) if a!=b)
        print("num of relabeled samples:", c)

        new_train_data = train_data.copy()
        new_train_data['target'] = new_labels


        train_dataset = MyBERTDataset(new_train_data, tokenizer)

        train_pred_probs = self.predict_probs(train_dataset)

        ordered_label_issues = find_label_issues(
            labels=train_data['target'],
            pred_probs=train_pred_probs,
            return_indices_ranked_by='self_confidence',
        )

        print("num of discarded samples:", len(ordered_label_issues))

        file_name_list = []
        for file in sorted(list(Path(self.config['preprocess_data_folder']).glob("*_train.csv")), key=lambda f:f.name):
            if "lower" in file.name or self.config["ascii_ratio"]+"_" not in file.name:continue
            data = pd.read_csv(file)

            new_data = data.copy()
            new_data['target'] = new_labels
            new_data = new_data.drop(ordered_label_issues)
            
            data.sort_values(by='ID').to_csv(f"{self.config['preprocess_data_folder']}/{remove_extension(file.name)}_original.csv", index=False)
            new_data.sort_values(by='ID').to_csv(file, index=False)

            file_name_list += [file.name]
        print("num of relabel_and_discard files:", len(file_name_list))
        print("relabel_and_discard files:", file_name_list)

        return cleanlab_relabel, new_labels


    def discard(self):
        model = self.model
        tokenizer = self.tokenizer

        model.eval()
        model.to(self.device)
        
        train_data = self.train_data
        train_dataset = MyBERTDataset(train_data, tokenizer)

        train_pred_probs = self.predict_probs(train_dataset)

        ordered_label_issues = find_label_issues(
            labels=train_data['target'],
            pred_probs=train_pred_probs,
            return_indices_ranked_by='self_confidence',
        )

        new_train_data = train_data.copy().drop(ordered_label_issues)

        train_data.sort_values(by='ID').to_csv(remove_extension(self.train_file_name) + '_original.csv', index=False)
        new_train_data.sort_values(by='ID').to_csv(self.train_file_name, index=False)

    def relabel(self):
        tokenizer = self.tokenizer
        cleanlab_relabel = self.cleanlab_relabel

        train_data = self.train_data
    
        new_labels = cleanlab_relabel.predict(MyBERTDataset(train_data, tokenizer))  # Get the new labels for relabeling
        print("num of new_labels:", len(new_labels))
        print("new_labels", new_labels)

        c = sum(1 for a ,b in zip(new_labels, train_data.target) if a!=b)
        print("num of relabeled samples:", c)

        new_train_data = train_data.copy()
        new_train_data['target'] = new_labels

        file_name_list = []
        for file in sorted(list(Path(self.config['preprocess_data_folder']).glob("*_train.csv")), key=lambda f:f.name):
            if "lower" in file.name or self.config["ascii_ratio"]+"_" not in file.name:continue
            data = pd.read_csv(file)

            new_data = data.copy()
            new_data['target'] = new_labels
            
            data.sort_values(by='ID').to_csv(f"{self.config['preprocess_data_folder']}/{remove_extension(file.name)}_original.csv", index=False)
            new_data.sort_values(by='ID').to_csv(file, index=False)

            file_name_list += [file.name]
        print("num of relabeled files:", len(file_name_list))
        print("relabeled files:", file_name_list)

        return cleanlab_relabel, new_labels
    
if __name__ == "__main__":

    import yaml

    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # %%
    print("clm")
    clm = CLM(config)
    clm.train()
    clm.validate()
    torch.cuda.empty_cache()
