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

class GemmaAugModel:
    def __init__(self, data):
        self.data = data
        self.model_id = "rtzr/ko-gemma-2-9b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    def get_few_shot_data(self, label_data, num):
        few_shot_data = label_data.sample(num,random_state=456)
        few_shot_list = [{"input":few_shot_data.iloc[i]["text"]} for i in range(len(few_shot_data))]
        return few_shot_list

    def prompt_few_shot(self, few_shot_list):
        system_message = """당신은 뉴스 기사 제목을 생성하는 어시스턴트입니다. 다음과 같은 지침을 따르세요.

        1. 주어진 예시와 같은 주제, 다른방식으로 새로운 뉴스 기사 제목을 생성하세요.
        2. 한줄 내외로 생성하세요.
        3. 이모티콘은 포함하지 마세요.
        4. 주어진 예시와 비슷한 내용은 생성하지 마세요.
        5. 제목의 톤은 중립적이며 뉴스 형식에 맞춰주세요.
        
        다음은 몇 가지 뉴스 기사 제목 예시입니다:

        """
        few_shot_messages = []
        for few_shot_data in few_shot_list:
            few_shot_messages.append(
                {"role":"user","content":"뉴스 기사 제목을 생성해주세요"}
            )
            few_shot_messages.append(
                {"role":"assistant","content":few_shot_data["input"]}
            )
        messages = [
            [{"role":"system", "content":system_message}] +
            few_shot_messages +
            [{"role":"user", "content":"뉴스 기사 제목을 생성해주세요"}]
        ]
        return messages

    def generate_sentence(self, few_shot_list):
        messages = self.prompt_few_shot(few_shot_list)
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
            do_sample=True,
            temperature=0.9,
            top_p=0.92,
        )
        decoded_data = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        decoded_data = decoded_data.replace("\n","")
        # few_shot_list.append({"input":decoded_data})
        print("GEN :",decoded_data)
        return decoded_data
    
    def inference_aug(self):

        label0_data = self.data[self.data["target"]==0]
        label1_data = self.data[self.data["target"]==1]
        label2_data = self.data[self.data["target"]==2]
        label3_data = self.data[self.data["target"]==3]
        label4_data = self.data[self.data["target"]==4]
        label5_data = self.data[self.data["target"]==5]
        label6_data = self.data[self.data["target"]==6]

        few_shot_list_0 = self.get_few_shot_data(label0_data, 40)
        few_shot_list_1 = self.get_few_shot_data(label1_data, 40)
        few_shot_list_2 = self.get_few_shot_data(label2_data, 40)
        few_shot_list_3 = self.get_few_shot_data(label3_data, 40)
        few_shot_list_4 = self.get_few_shot_data(label4_data, 40)
        few_shot_list_5 = self.get_few_shot_data(label5_data, 40)
        few_shot_list_6 = self.get_few_shot_data(label6_data, 40)

        lists = [few_shot_list_0,few_shot_list_1,few_shot_list_2,
                 few_shot_list_3,few_shot_list_4,few_shot_list_5,few_shot_list_6]

        generated_data = []
        for i, few_shots in enumerate(lists):
            for _ in range(100):
                gen_data = self.generate_sentence(few_shots)
                generated_data.append([gen_data, i])
        
        gen_data = pd.DataFrame()
        gen_texts = []
        gen_targets = []

        for gen_datas in generated_data:
            gen_text, gen_target = gen_datas
            gen_texts.append(gen_text)
            gen_targets.append(gen_target)

        aug_prefix = "aug-v1_gem_train_"
        aug_prefix_id = []
        for i in range(len(gen_texts)):
            aug_id = aug_prefix+str(i)
            aug_prefix_id.append(aug_id)

        gen_data["ID"] = aug_prefix_id
        gen_data["text"] = gen_texts
        gen_data["target"] = gen_targets

        gen_data = gen_data.sample(frac=1).reset_index(drop=True)
        print("Complete_gemma_augmentation")
        return gen_data