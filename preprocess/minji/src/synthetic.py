import os
import logging
import pandas as pd
from tqdm import tqdm
from llama import Llama
from typing import List
import torch
logging.basicConfig(level="INFO")

class SyntheticData():
    def __init__(self, system_prompt_path):
        self.system_prompt_path = system_prompt_path

    def generate_data(self, target_range, save_path, fewshot_data=None, fewshot_path=None, n_random_fewshots=30, num=100):
        # 폴더가 없으면 생성
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        all_data = []

        for target in target_range:
            llama = Llama()

            for idx in tqdm(range(num)):
                ID = f"synthetic_{target}_{idx:04}"
                input_text = "뉴스 기사 제목을 생성하라"

                # fewshot_path를 사용할 경우 각 target에 맞는 fewshot 파일을 로드
                if fewshot_path:
                    fewshot_file = f"{fewshot_path}_{target}.json"
                    text = llama.inference(self.system_prompt_path, input_text, fewshot_path=fewshot_file)
                else:
                    text = llama.inference(self.system_prompt_path, input_text, fewshot_data=fewshot_data, n_random_fewshots=n_random_fewshots)

                all_data.append({"ID": ID, "text": text, "target": target})

            logging.info(f"Target {target} generated")

            del llama.model, llama.tokenizer, llama
            torch.cuda.empty_cache()

        # 전체 데이터 DataFrame으로 변환 후 저장
        synthetic_data = pd.DataFrame(all_data)
        synthetic_data.to_csv(f"{save_path}synthetic.csv", index=False)
        logging.info("Saved synthetic data")

    def generate_all_targets(self, save_path, fewshot_data, n_random_fewshots=30, num=100):
        self.generate_data(target_range=range(7), save_path=save_path, fewshot_data=fewshot_data, n_random_fewshots=n_random_fewshots, num=num)

    def generate_targetwise(self, targets: List, fewshot_path, save_path, num=200):
        self.generate_data(target_range=targets, save_path=save_path, fewshot_path=fewshot_path, num=num)
