import pandas as pd
from tqdm import tqdm
from llama import LlamaInference
import logging
import torch

train = pd.read_csv("../../datasets/v0.0.6/train.csv")
valid = pd.read_csv("../../datasets/v0.0.6/validation.csv")
data = pd.concat([train, valid], axis=0)


model_name = "meta-llama/Llama-3.1-8B-Instruct"
data_path = "../../datasets/v0.0.6/"
prompt_path = "./prompt/title_generation/generation_prompt_v1.txt"
fewshot_sample=True
previous_num = 200
generation_num = 300
saving_path = "../../datasets/v1.1.1/"
previous_syn_path = "../../datasets/v1.0.1/"

for target in range(7):
    inferencer = LlamaInference(model_name, data_path, prompt_path, target, fewshot_sample)

    data = {"ID": [], "text": [], "target": []}

    for i in tqdm(range(generation_num)):
        idx = previous_num + i
        ID = 'synthetic_llama_' + str(target) + f"{idx:04}"
        text = inferencer.inference("뉴스 기사 제목을 생성하라")
        data["ID"].append(ID)
        data["text"].append(text)
        data["target"].append(target)

    synthetic_data = pd.DataFrame(data)
    synthetic_data.to_csv(f"{saving_path}synthetic_{target}.csv", index=False)
    logging.info(f"Target {target} saved")

    # inferencer.model.to("cpu")
    # inferencer.tokenizer.to("cpu")
    del inferencer.model, inferencer.tokenizer, inferencer 
    torch.cuda.empty_cache()

# 데이터 합치기
logging.info(f"Concatenating synthetic data")
org_train = pd.read_csv(f"{data_path}train.csv")
org_train = org_train[['ID', 'text', 'target']]
prev_syn = pd.read_csv(f"{previous_syn_path}synthetic.csv")

syn_0 = pd.read_csv(f"{saving_path}synthetic_0.csv")
syn_1 = pd.read_csv(f"{saving_path}synthetic_1.csv")
syn_2 = pd.read_csv(f"{saving_path}synthetic_2.csv")
syn_3 = pd.read_csv(f"{saving_path}synthetic_3.csv")
syn_4 = pd.read_csv(f"{saving_path}synthetic_4.csv")
syn_5 = pd.read_csv(f"{saving_path}synthetic_5.csv")
syn_6 = pd.read_csv(f"{saving_path}synthetic_6.csv")

new_syn = pd.concat([syn_0, syn_1, syn_2, syn_3, syn_4, syn_5, syn_6], axis=0).reset_index(drop=True)
syn = pd.concat([prev_syn, new_syn], axis=0).reset_index(drop=True)
train = pd.concat([org_train, syn], axis=0).reset_index(drop=True)

syn.to_csv(f"{saving_path}synthetic.csv", index=False)
train.to_csv(f"{saving_path}train.csv", index=False)