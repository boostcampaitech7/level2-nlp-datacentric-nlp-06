import pandas as pd
from tqdm import tqdm
from llama import LlamaInference

model_name = "meta-llama/Llama-3.1-8B-Instruct"
data_path = "../../datasets/v0.0.6/"
prompt_path = "./prompt/title_generation/generation_prompt_v1.txt"
fewshot_sample = False
examples_path = "./prompt/title_generation/generation_few_shot_2_v1.json"
previous_num = 400
generation_num = 100
previous_syn_path = "../../datasets/v2.1.4/"
saving_path = "../../datasets/v2.1.5/"

target = 2

inferencer = LlamaInference(model_name, data_path, prompt_path, target, fewshot_sample, examples_path)
data = {"ID": [], "text": [], "target": []}

for i in tqdm(range(generation_num)):
    idx = previous_num + i
    ID = 'synthetic_target_' + str(target) + f"{idx:04}"
    text = inferencer.inference("뉴스 기사 제목을 생성하라")
    data["ID"].append(ID)
    data["text"].append(text)
    data["target"].append(target)

synthetic_data = pd.DataFrame(data)

# 데이터 합치기
org_train = pd.read_csv(f"{data_path}train.csv")
prev_syn = pd.read_csv(f"{previous_syn_path}synthetic_{target}.csv")
syn = pd.concat([prev_syn, synthetic_data], axis=0).reset_index(drop=True)
syn.to_csv(f"{saving_path}synthetic_{target}.csv", index=False)

train = pd.concat([org_train, syn], axis=0).reset_index(drop=True)
train.to_csv(f"{saving_path}train.csv", index=False)







# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# data_path = "../../datasets/v0.0.6/"
# prompt_path = "./prompt/title_generation/generation_prompt_v1.txt"
# fewshot_sample = False
# examples_path = "./prompt/title_generation/generation_few_shot_4_v1.json"
# previous_num = 0
# generation_num = 100
# saving_path = "../../datasets/v2.3.1/"

# target = 5

# inferencer = LlamaInference(model_name, data_path, prompt_path, target, fewshot_sample, examples_path)
# data = {"ID": [], "text": [], "target": []}

# for i in tqdm(range(generation_num)):
#     idx = previous_num + i
#     ID = 'synthetic_target_' + str(target) + f"{idx:04}"
#     text = inferencer.inference("뉴스 기사 제목을 생성하라")
#     data["ID"].append(ID)
#     data["text"].append(text)
#     data["target"].append(target)

# synthetic_data = pd.DataFrame(data)

# # 데이터 합치기
# org_train = pd.read_csv(f"{data_path}train.csv")
# synthetic_data.to_csv(f"{saving_path}synthetic_{target}.csv", index=False)

# train = pd.concat([org_train, synthetic_data], axis=0).reset_index(drop=True)
# train.to_csv(f"{saving_path}train.csv", index=False)
