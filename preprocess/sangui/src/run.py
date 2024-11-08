# %% [markdown]
# # Data-Centric NLP 대회: 주제 분류 프로젝트

# %%
import os
import random
import yaml

import numpy as np
import torch

from ascii import *
from mlm import MLM
from btm import BTM
from rlm import RLM
from clm import CLM
from blm import BLM
from utils import *



## for wandb setting
os.environ['WANDB_DISABLED'] = 'true'

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



with open('../config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

pivot_name = config["pivot_name"]
ascii_ratio = config["ascii_ratio"]

train_file_name = config["data_folder"]+'/train.csv'


# %%
print("ascii_filter")
ascii = Ascii(config)

ascii.ascii_filter(train_file_name, pivot_name, ascii_ratio)

# %%
print("mlm")
mlm = MLM(config)
mlm.train()
mlm.test()
torch.cuda.empty_cache()

# %%
print("btm")
btm = BTM(config)
# btm.train() # BTM train not implemented
btm.test()
torch.cuda.empty_cache()

# %%
print("text data split")
ascii.text_train_valid_split(pivot_name, ascii_ratio)

# %%
print("rlm")
rlm = RLM(config)
rlm.train()
rlm.test()
torch.cuda.empty_cache()

# %%
print("clm")
clm = CLM(config)
clm.train()
clm.test()
torch.cuda.empty_cache()

# %%
print("rlm")
rlm = RLM(config)
rlm.train()
rlm.test()
torch.cuda.empty_cache()

# %%
print("label split")
ascii.label_train_valid_split(pivot_name, ascii_ratio)

print("train valid merge")
ascii.train_valid_merge(pivot_name, ascii_ratio)

# %%
print("blm")
blm = BLM(config)
blm.train()
blm.test()
torch.cuda.empty_cache()