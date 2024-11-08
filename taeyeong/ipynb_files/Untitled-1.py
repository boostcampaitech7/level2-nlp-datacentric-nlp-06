# %%
from huggingface_hub import login
import transformers
import torch
import pandas as pd
from tqdm import tqdm

# %%
MY_HF_TOKEN = "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"
login(token=MY_HF_TOKEN)


# %%
dataset = pd.read_csv("/data/ephemeral/home/sty/data/train.csv")
clean_df = pd.read_csv("/data/ephemeral/home/sty/data/clean_data.csv")

# %%
keys = {0: "정치", 1: "스포츠", 2: "경제", 3: "사회", 4: "문화", 5: "IT/과학", 6: "국제"}
keys_str = []
for idx, key in list(keys.items()):
    keys_str.append(f'{idx}: {key}')
keys_str = ', '.join(keys_str)

print(keys_str)

# %%
len(clean_df)

# %%


# %%
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

keys = {0: "정치", 1: "스포츠", 2: "경제", 3: "사회", 4: "문화", 5: "IT/과학", 6: "국제"}
keys_str = []
for idx, key in list(keys.items()):
    keys_str.append(f'{idx}: {key}')
keys_str = ', '.join(keys_str)

#print(keys_str)

keywords = []

PROMPT = f'''
    당신은 기사 제목을 보고 어떤 분야의 기사인지 맞추는 전문가입니다.
    기사의 분야에 대한 정수 값은 {keys_str}와 같습니다.
    기사를 보고 해당하는 분야의 정수 값을 리턴해주세요.
    숫자만 리턴해주세요.
  '''
print(PROMPT)
    
for i in tqdm(range(len(clean_df))):
    #data = ', '.join(sam_df['text'] + ':' + sam_df['target'].astype(str))
    data = clean_df.iloc[i]["text"]

    messages = [{"role": "system", "content": f"{PROMPT}"}] + \
            [{"role": "user", "content": f"{data}"}]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    keywords.append(outputs[0]["generated_text"][len(prompt):])
print(keywords)

# %%
keywords1 = keywords

# %%
len(keywords1)

# %%
len(clean_df)

# %%
clean_df["new_target"] = keywords1
clean_df.head()

# %%
import torch, gc
gc.collect()
torch.cuda.empty_cache()

# %%
model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map=DEVICE,
)

pipeline.model.eval()

keys = {0: "정치", 1: "스포츠", 2: "경제", 3: "사회", 4: "문화", 5: "IT/과학", 6: "국제"}
keys_str = []
for idx, key in list(keys.items()):
    keys_str.append(f'{idx}: {key}')
keys_str = ', '.join(keys_str)

#print(keys_str)

keywords = []

PROMPT = f'''
    당신은 기사 제목을 보고 어떤 분야의 기사인지 맞추는 전문가입니다.
    기사의 분야에 대한 정수 값은 {keys_str}와 같습니다.
    기사를 보고 해당하는 분야의 정수 값을 리턴해주세요.
    결과 값에 숫자만 있게 해주세요. 글자 하나 없이 오로지 숫자만
  '''
print(PROMPT)

for i in tqdm(range(len(dataset))):
    data = dataset.iloc[i]["text"]

    messages = [{"role": "system", "content": f"{PROMPT}"}] + \
            [{"role": "user", "content": f"{data}"}]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    key = outputs[0]["generated_text"][len(prompt):]
    
    keywords.append(int(key))
print(keywords)
dataset["new_target"] = keywords

# %%
dataset.head()

# %%
dataset.to_csv("/data/ephemeral/home/sty/data/relabel_train.csv")

# %%
text_datas = dataset["text"]
def calculate_ascii_noise(text):
    total_chars = len(text)
    ascii_chars = 0 
    for ch in text:
        if ch == " ":
            continue
        if ord(ch) < 128:
            ascii_chars+=1
            
    non_ascii_chars = total_chars - ascii_chars  # 비-ASCII 문자 개수
    
    ascii_ratio = (ascii_chars / total_chars) * 100  # ASCII 문자 비율 (%)
    
    return ascii_chars, non_ascii_chars, ascii_ratio
asc_counts = []; non_asc_counts = []; percentages = []
# ASCII 문자 개수, 비-ASCII 문자 개수, ASCII 비율 계산
for text_with_noise in text_datas:
    ascii_count, non_ascii_count, ascii_percentage = calculate_ascii_noise(text_with_noise)
    ascii_percentage = round(ascii_percentage,2)
    asc_counts.append(ascii_count)
    non_asc_counts.append(non_ascii_count)
    percentages.append(ascii_percentage)
    
print(len(asc_counts), len(non_asc_counts), len(percentages))
dataset["ratio"] = percentages

# %%
dataset.head()

# %%
new_dataset = dataset[dataset['ratio'] < 40]
new_dataset.head()

# %%
len(new_dataset)

# %%
new_dataset['target'] = new_dataset['new_target']
new_dataset = new_dataset.drop(columns=['new_target'])
new_dataset.head()

# %%
new_dataset.to_csv("/data/ephemeral/home/sty/data/relabel_translator_train.csv")


