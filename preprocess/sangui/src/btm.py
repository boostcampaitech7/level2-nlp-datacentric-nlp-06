# %% [markdown]
# # Data-Centric NLP 대회: 주제 분류 프로젝트

# %%
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer
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



class BTM():
    def __init__(self, config):
        self.config = config

        pivot_name = config["pivot_name"]
        ascii_ratio = config["ascii_ratio"]

        self.device = config["device"]

        self.model_name = config["btm"]["model_name"]
        print("btm model_name", self.model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self.batch_size = config["btm"]["batch_size"]
        
        self.train_file_name = None
        self.test_file_name = f"{pivot_name}_higher_{ascii_ratio}_mlm.csv"
        self.train_data = None
        self.test_data = pd.read_csv(config["preprocess_data_folder"]+self.test_file_name)

    def train(self):
        pass

    def test(self, data=None):
        
        batch_size = self.batch_size

        tokenizer = self.tokenizer
        model = self.model

        data = self.test_data if data is None else data

        bt_sentences = []
        num_samples = len(data.text)
        for i in range(num_samples//batch_size + min(1, num_samples%batch_size)):
            sentences = data.text[i*batch_size:(i+1)*batch_size].to_list()
            sentences = [remove_repeated_characters(sentence) for sentence in sentences]

            inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

            model.to(self.device)
            inputs.to(self.device)
            with torch.no_grad():
                translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("kor_Hang"), max_length=100)
                translated_sentences = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            #translated_sentences = [remove_repeated_characters(sentence) for sentence in translated_sentences]

            #for j in range(i*batch_size, min(num_samples, (i+1)*batch_size)):
            #    data[j, 'text'] = translated_sentences[j%batch_size]

            bt_sentences += [remove_repeated_characters(sentence) for sentence in translated_sentences]

        data.text = bt_sentences

        file_name = remove_extension(self.test_file_name)
        data.sort_values(by='ID').to_csv(self.config["preprocess_data_folder"]+file_name+"_btm.csv", index=False)

    def print(self, data=None):
        tokenizer = self.tokenizer
        model = self.model
        data = self.test_data if data is None else data
        
        for i in range(len(data.text)):
            inputs = tokenizer(data.text[i], return_tensors='pt')
        
            model.to(self.device)
            inputs.to(self.device)
        
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("kor_Hang"), max_length=100)
            print(data.text[i])
            print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
            print()
