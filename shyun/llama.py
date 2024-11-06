import os, json, tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import prompt

MODEL = 'Bllossom/llama-3.2-Korean-Bllossom-3B' # 'beomi/Llama-3-Open-Ko-8B'

class Llama:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            device_map='auto'
        )

        self.model.eval()

        self.terminators = [
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def generate(self, message):
        input_ids = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors='pt'
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    def extract_label(self, train_dataset):
        sys_prompt, fewshot = prompt.extract_label_prompt()

        target = train_dataset[(0.30<train_dataset['polluted_lv']) & (train_dataset['polluted_lv']<0.40)]

        keywords = []
        for i in range(7):
            keys = []
            for j in range(10):
                data = target[(target['target']==i)].sample(5)['text'].values
                data = '\n'.join(data)

                messages = \
                    [{"role": "system", "content": f"{sys_prompt}"}] + \
                    fewshot + \
                    [{"role": "user", "content": f"{data}"}]
                
                result = self.generate(messages)

                keys.append(result)
            keywords.append(keys)
        
        sys_prompt = '''제공된 데이터들은 뉴스 기사 분야 10개에 대한 단어입니다. 키워드를 추출하세요.
- 많이 등장하는 단어보다 전체 단어들의 맥락을 고려하세요.
- 여러 개의 키워드가 있다면 가장 포괄적인 키워드 한 개만 출력하세요.'''

        final_keys = []
        for keys in keywords:
            keys = ', '.join(keys)
            messages = \
                [{"role": "system", "content": sys_prompt}] + \
                fewshot +\
                [{"role": "user", "content": keys}]
            
            result = self.generate(messages)
            final_keys.append(result)
        
        key_maps = {}
        for idx, key in enumerate(final_keys):
            key_maps[idx] = key
        
        with open('./key_maps.json', 'w') as f:
            json.dump(key_maps, f, ensure_ascii=False)

    def clean_label(self, keys, train_dataset, p, path): # keys: json 형태의 key 그대로 전달
        sys_prompt, fewshot = prompt.clean_label_prompt(keys)

        target = train_dataset[train_dataset['polluted_lv'] < p]

        cnt = 0
        for _, data in tqdm(target.iterrows(), desc='labeling', total=len(target)):
            messages = \
                [{"role": "system", "content": sys_prompt}] + \
                fewshot + \
                [{"role": "user", "content": data['text']}]
            
            result = self.generate(messages)

            if result != '불가':
                cnt += 1
                train_dataset.loc[train_dataset['ID']==data['ID'], 'target'] = int(result)
        
        print(f"총 {cnt}개의 라벨이 다시 설정되었습니다.")
        train_dataset.to_csv(os.path.join(path, 'label_cleaned.csv'), index=False)

    def clean_text(self, keys, train_dataset, p, path):
        target = train_dataset[train_dataset['pollute_lv'] > p]

        for idx, key in tqdm(enumerate(keys.values()), total=len(keys), position=0):
            sys_prompt, fewshot = prompt.clean_text_prompt(key)

            key_target = target[target['target']==idx]
            for _, data in tqdm(key_target.iterrows(), desc=f"label={idx}", total=len(key_target), position=1, leave=False):
                messages = \
                    [{"role": "system", "content": sys_prompt}] + \
                    fewshot + \
                    [{"role": "user", "content": data['text']}]
                
                result = self.generate(messages)
                train_dataset.loc[train_dataset['ID']==data['ID'], 'text'] = result
        
        train_dataset.to_csv(os.path.join(path, 'text_cleaned.csv'), index=False)
