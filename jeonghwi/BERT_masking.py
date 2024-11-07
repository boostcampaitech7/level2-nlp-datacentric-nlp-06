import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

import evaluate
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MLM_Filter:
    def __init__(self, data):
        self.data = data
        self.model_id = "jian1114/jian_KoBART_title"
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        self.model = BartForConditionalGeneration.from_pretrained(model_id)

        # Masking
        self.text_masking()

    def replace_ascii_with_mask(self, text):
            # 결과를 저장할 리스트
            result = []
            
            # 입력 문자열을 한 문자씩 순회
            for char in text:
                # 공백은 그대로 추가
                if char == ' ':
                    result.append(char)
                # ASCII 코드 범위(0 ~ 127) 확인, 공백 제외
                elif ord(char) <= 127:
                    # ASCII 문자면 <mask>로 대체
                    result.append('<mask>')
                else:
                    # 그렇지 않으면 원래 문자를 그대로 추가
                    result.append(char)
            
            # 리스트를 문자열로 변환하여 반환
            return ''.join(result)

    def text_masking(self, text_datas):
        # ascii code에 해당하는 부분에 Mask 처리
        masked_text = [self.replace_ascii_with_mask(text) for text in text_datas]
        return masked_text
    
    def BART_inference(self):
        masked_text = self.text_masking(self.data["text"])
        input_ids = self.tokenizer(masked_text, return_tensors='pt')['input_ids']
        
        result_text = []
        for inputs in input_ids:
            output_ids = self.model.generate(
                input_ids, 
                max_length=50, 
                early_stopping=True,
                num_beams=3,
                no_repeat_ngram_size=2,  # 3-그램 이상의 반복 방지
                temperature=0.7,         # 다양성 조절
                top_p=0.92,
                top_k=50
            )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            result_text.append(output_text)
        
        self.data["text"] = result_text
        return self.data