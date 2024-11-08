import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

import evaluate
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

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

class GemmaFilterModel:
    def __init__(self, data):
        self.data = data
        self.model_id = "rtzr/ko-gemma-2-9b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def get_few_shot_filter(self):
        few_shot_list = [
            {
                "input":"서울문고 N풍그U에 인수… f라인서점 교m영풍y양강d도로",
                "output":"서울문고, 영풍그룹에 인수…오프라인서점 교보-영풍 양강구도로"
            },
            {
                "input":"프로야구~롯TKIAs광주 경기 y천취소",
                "output":"프로야구 롯데-KIA 광주 경기 우천취소"
            },
            {
                "input":"러시아 시리아서 S500 발사했나…러 국방부는 부인",
                "output":"러시아 시리아서 S500 발사했나…러 국방부는 부인"
            },
            {
                "input":"xW리 a)엔 예비후0V사전여론조사 결과 유출 논c",
                "output":"새누리, 이번엔 예비후보 사전여론조사 결과 유출 논란"
            },
            {
                "input":"조국U압수수색f검2와 통7 파장…한h·바(미래 &핵K진i",
                "output":"조국 '압수수색 검사와 통화' 파장…한국·바른미래 '탄핵추진'"
            }
        ]
        few_shot_messages = []
        for few_shot_data in few_shot_list:
            few_shot_messages.append(
                {"role":"user","content":few_shot_data["input"]}
            )
            few_shot_messages.append(
                {"role":"assistant","content":few_shot_data["output"]}
            )
        return few_shot_messages

    
    def generate_sentence(self, messages, inp_text):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(DEVICE)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.6,
            top_p=0.9,
        )
        decoded_data = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print("RAW :",inp_text)
        print("GEN :",decoded_data)
        return decoded_data
    
    def inference_filter(self):
        system_message = """당신은 뉴스 기사 제목에 있는 노이즈를 제거하는 어시스턴트로, 한 문장 내외의 뉴스 기사 제목을 복원해야합니다. 다음 지침을 따르세요.

        1. 제목에 오류가 없다면 수정하지 마세요.
        2. 원본 제목의 의미, 뉘앙스, 형식을 최대한 유지하세요.
        3. 오타, 특수문자 오용, 인코딩 오류 등을 수정하세요.
        4. 제목이 이해하기 어렵다면 명확하게 다듬으세요.
        """
        generate_data = []
        few_shot_messages =self.get_few_shot_filter()
        text_datas = self.data["text"]

        for inp_text in text_datas:
            messages = [
                [{"role":"system", "content":system_message}] +
                few_shot_messages +
                [{"role":"user", "content":inp_text}]
            ]
            decoded_data = self.generate_sentence(messages, inp_text).replace("\n","")
            generate_data.append(decoded_data)
        
        self.data["text"] = generate_data

        return self.data