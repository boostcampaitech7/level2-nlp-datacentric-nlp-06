import os
import random

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split

from utils import *



## for wandb setting
os.environ['WANDB_DISABLED'] = 'true'

# %% [markdown]
# ## Set Hyperparameters

# %%
SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



class MLM():
    def __init__(self, config):
        self.config = config

        self.output_dir = config["mlm"]["model_output_dir"]
        
        self.model_name = config["mlm"]["model_name"] 
        print("mlm model_name", self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        self.device = config["device"]

        self.batch_size = config["mlm"]["batch_size"]

        pivot_name = config["pivot_name"]
        ascii_ratio = config["ascii_ratio"]
        self.train_file_name = f"{pivot_name}_lower_{ascii_ratio}.csv"
        self.test_file_name = f"{pivot_name}_higher_{ascii_ratio}.csv"
        self.train_data = pd.read_csv(config["preprocess_data_folder"]+self.train_file_name)
        self.test_data = pd.read_csv(config["preprocess_data_folder"]+self.test_file_name)


        tokenizer = self.tokenizer
        model = self.model

        batch_size = self.batch_size

        data_train, data_valid = train_test_split(self.train_data, test_size=0.3, random_state=SEED)

        dataset_train = BERTDataset(data_train, tokenizer)
        dataset_valid = BERTDataset(data_valid, tokenizer)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="steps",
            save_strategy="no",
            learning_rate=2e-5,
            num_train_epochs=2,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            seed=SEED
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            data_collator=data_collator,
            tokenizer=tokenizer
        )

    def train(self):
        self.trainer.train()
        self.validate()
        self.trainer.save_model(self.output_dir)

    def validate(self):
        print(self.trainer.evaluate())

    def test(self):
        tokenizer = self.tokenizer
        model = self.model
        device =self.device

        model.eval()
        model.to(device)

        data = self.test_data
        for i in range(len(data.text)):
            data.loc[i, 'text'] = replace(r"[\x00-\x7F]", r"[MASK]", data.text[i])

        dataset_all = MyBERTDataset(data, tokenizer)
        for i, batch in enumerate(dataset_all):
            with torch.no_grad():
                batch['input_ids'] = batch['input_ids'].to(device)
                batch['attention_mask'] = batch['attention_mask'].to(device)
                batch['labels'].to(device)
                del batch['labels']
                outputs = model(**batch)
                predictions = outputs.logits

            data.loc[i, 'text'] = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(torch.argmax(predictions[0], dim=-1))[1:-1])

        file_name = remove_extension(self.test_file_name)
        data.sort_values(by='ID').to_csv(self.config["preprocess_data_folder"]+file_name+"_mlm.csv", index=False)
