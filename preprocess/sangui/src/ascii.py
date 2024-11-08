from pathlib import Path
import random
import numpy as np
import pandas as pd

import torch
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

class Ascii():
    def __init__(self, config):
        self.config = config
        
        Path(self.config['preprocess_data_folder']).mkdir(parents=True, exist_ok=True)
        #Path(self.config['train_data_folder']).mkdir(parents=True, exist_ok=True)
        #Path(self.config['valid_data_folder']).mkdir(parents=True, exist_ok=True)

    def ascii_remove(self, file_name, pivot_name, ascii_ratio):
        data = pd.read_csv(file_name)

        data['ascii_ratio'] = data.text.map(calculate_ascii_ratio)

        #df_ascii = data[(data.ascii_ratio >= float(ascii_ratio)) & (data.ascii_ratio <= 70)].copy()
        df_ascii = data[data.ascii_ratio >= float(ascii_ratio)].copy()
        df_not_ascii = data[data.ascii_ratio < float(ascii_ratio)].copy()

        df_ascii['text'] = df_ascii.text.map(remove_not_korean_without_space)

        print(ascii_ratio)
        print(len(df_ascii.text))
        print(len(df_not_ascii.text))


        df_ascii.sort_values(by='ID').to_csv(f"{self.config['preprocess_data_folder']}/{pivot_name}_higher_{ascii_ratio}.csv", index=False)
        df_not_ascii.sort_values(by='ID').to_csv(f"{self.config['preprocess_data_folder']}/{pivot_name}_lower_{ascii_ratio}.csv", index=False)



    def ascii_filter(self, file_name, pivot_name, ascii_ratio):
        data = pd.read_csv(file_name)

        data['ascii_ratio'] = data.text.map(calculate_ascii_ratio)

        #df_ascii = data[(data.ascii_ratio >= float(ascii_ratio)) & (data.ascii_ratio <= 70)].copy()
        df_ascii = data[data.ascii_ratio >= float(ascii_ratio)].copy()
        df_not_ascii = data[data.ascii_ratio < float(ascii_ratio)].copy()

        print(ascii_ratio)
        print(len(df_ascii.text))
        print(len(df_not_ascii.text))


        df_ascii.sort_values(by='ID').to_csv(f"{self.config['preprocess_data_folder']}/{pivot_name}_higher_{ascii_ratio}.csv", index=False)
        df_not_ascii.sort_values(by='ID').to_csv(f"{self.config['preprocess_data_folder']}/{pivot_name}_lower_{ascii_ratio}.csv", index=False)

    def text_train_valid_split(self, pivot_name, ascii_ratio):
        text_error_name = f"{self.config['preprocess_data_folder']}/{pivot_name}_higher_{ascii_ratio}"

        data = pd.read_csv(text_error_name+".csv")
        data_train, data_valid = train_test_split(data, test_size=0.3, random_state=SEED)
        data_train.sort_values(by='ID').to_csv(text_error_name+"_train.csv", index=False)
        data_valid.sort_values(by='ID').to_csv(text_error_name+"_valid.csv", index=False)

        data = pd.read_csv(text_error_name+"_mlm.csv")
        data_train, data_valid = train_test_split(data, test_size=0.3, random_state=SEED)
        data_train.sort_values(by='ID').to_csv(text_error_name+"_mlm_train.csv", index=False)

        data = pd.read_csv(text_error_name+"_mlm_btm.csv")
        data_train, data_valid = train_test_split(data, test_size=0.3, random_state=SEED)
        data_train.sort_values(by='ID').to_csv(text_error_name+"_mlm_btm_train.csv", index=False)
        #data_valid.to_csv(text_error_name+"_mlm_btm_valid.csv", index=False)

    def label_train_valid_split(self, pivot_name, ascii_ratio):
        label_error_name = f"{self.config['preprocess_data_folder']}/{pivot_name}_lower_{ascii_ratio}"
        
        data_origin = pd.read_csv(label_error_name+".csv")
        data_relabeled = pd.read_csv(f"{label_error_name}_rlm.csv")

        c=0
        for a,b in zip(data_origin.target, data_relabeled.target):
            if a==b:c+=1
        print("num of original_label & relabel:", c)

        data_labeling_error = data_relabeled[data_relabeled.target != data_origin.target]
        data_train, data_valid = train_test_split(data_labeling_error, test_size=0.3, random_state=SEED)
        data_train.sort_values(by='ID').to_csv(f"{label_error_name}_rlm_train.csv", index=False)
        data_valid.sort_values(by='ID').to_csv(f"{label_error_name}_rlm_valid.csv", index=False)

        data_normal_candidates = data_relabeled[data_relabeled.target == data_origin.target]
        data_train, data_valid = train_test_split(data_normal_candidates, test_size=0.3, random_state=SEED)
        data_normal_candidates.copy().sort_values(by='ID').to_csv(f"{label_error_name}_rlm_normal_candidates.csv", index=False)
        data_train.sort_values(by='ID').to_csv(f"{label_error_name}_rlm_normal_candidates_train.csv", index=False)
        data_valid.sort_values(by='ID').to_csv(f"{label_error_name}_rlm_normal_candidates_valid.csv", index=False)

    def label_merge(self, pivot_name, ascii_ratio):
        label_error_name = f"{self.config['preprocess_data_folder']}/{pivot_name}_lower_{ascii_ratio}"

        data1 = pd.read_csv(f"{label_error_name}_rlm_normal_candidates_train.csv")
        data2 = pd.read_csv(f"{label_error_name}_rlm_normal_candidates_valid.csv")

        data = pd.concat([data1, data2])
        data.sort_values(by='ID').to_csv(f"{label_error_name}_rlm_normal_candidates.csv", index=False)

    def train_valid_merge(self, pivot_name, ascii_ratio):
        data_list = []
        file_name_list = []
        for file in sorted(list(Path(self.config['preprocess_data_folder']).glob("*_train.csv")), key=lambda f:f.name):
            if pivot_name not in file.name or ascii_ratio + "_" not in file.name:continue
            data = pd.read_csv(file)
            data_list += [data]
            file_name_list += [file.name]
        print("num of train data files:", len(data_list))
        print("merge train data files:", file_name_list)
        train_data = pd.concat(data_list)
        #train_data.to_csv(self.config['data_folder']+"/train.csv", index=False)

        data_list = []
        file_name_list = []
        for file in sorted(list(Path(self.config['preprocess_data_folder']).glob("*_valid.csv")), key=lambda f:f.name):
            if pivot_name not in file.name or ascii_ratio + "_" not in file.name:continue
            data = pd.read_csv(file)
            data_list += [data]
            file_name_list += [file.name]
        print("num of valid data files:", len(data_list))
        print("merge valid data files:", file_name_list)
        valid_data = pd.concat(data_list)
        #valid_data.to_csv(self.config['data_folder']+"/valid.csv", index=False)

        data = pd.concat([train_data, valid_data])
        data.to_csv(self.config['data_folder']+"/new_train.csv", index=False)