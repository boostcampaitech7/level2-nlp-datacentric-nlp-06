import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import json


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



if __name__ == "__main__":
    
    # load data

    # Noise ratio 30% under data

    # 1. Filter Select    
    # BERT Mask Filter option (25% 이상 데이터)

    # Gemma Filter option (30% 아래 데이터)

    # 2. SBERT Relabeling

    # 3. Augmentation Select
    # Back Translation

    # Gemma Prompting