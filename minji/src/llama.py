import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import json

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
class Llama:
    def __init__(self):
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # load model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False


    def inference(self, system_prompt_path, input_text, fewshot_path=None, fewshot_data=None, n_random_fewshots=30):
        # 시스템 프롬프트 설정
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
        messages = [{"role": "system", "content": system_prompt}]
        
        # few-shot 설정
        if fewshot_path:
            # 고정된 few-shot
            with open(fewshot_path, "r") as f:
                fewshots = json.load(f)["few-shots"]
            messages.extend(fewshots)
        elif fewshot_data is not None:
            # 랜덤 few-shot
            sampled_fewshots = fewshot_data.sample(n=n_random_fewshots)['text']
            for shot in sampled_fewshots:
                messages.append({"role": "user", "content": "뉴스 기사 제목을 생성하라"})
                messages.append({"role": "assistant", "content": shot})
        
        # 사용자 입력 설정
        messages.append({"role": "user", "content": input_text})
        
        # 모델 입력 토큰화
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # 모델 추론
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=64,
                do_sample=True,
                pad_token_id=pad_token_id)
        
        # 새로 생성된 토큰만 추출
        new_tokens = outputs[0][inputs.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
