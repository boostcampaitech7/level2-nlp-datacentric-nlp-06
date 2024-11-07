import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import json

import evaluate
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoModelForCausalLM
)
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Seed Set
SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# 디바이스 설정 (GPU가 사용 가능하면 GPU를 사용하고, 그렇지 않으면 CPU 사용)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

