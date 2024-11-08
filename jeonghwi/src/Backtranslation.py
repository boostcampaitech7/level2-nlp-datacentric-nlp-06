from transformers import M2M100ForConditionalGeneration, NllbTokenizer
import pandas as pd

class BTModel:
    def __init__(self, data):
        self.data = data
    def translate(seelf, text, model, tokenizer, max_length=512):
        # 입력 문장을 토큰화
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        # print(inputs["input_ids"])
        # 모델을 사용하여 번역
        translated = model.generate(**inputs)
        
        # 번역된 결과를 텍스트로 디코딩
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
    
    def backtranslation(self):
        # kor to en
        ids = []
        aug_t = []
        targets = []
        # 한국어 → 영어 번역 모델
        ko_to_en_model_name = 'NHNDQ/nllb-finetuned-ko2en'
        ko_to_en_model = M2M100ForConditionalGeneration.from_pretrained(ko_to_en_model_name)
        ko_to_en_tokenizer = NllbTokenizer.from_pretrained(ko_to_en_model_name)

        # 영어 → 한국어 번역 모델
        en_to_ko_model_name = 'NHNDQ/nllb-finetuned-en2ko'
        en_to_ko_model = M2M100ForConditionalGeneration.from_pretrained(en_to_ko_model_name)
        en_to_ko_tokenizer = NllbTokenizer.from_pretrained(en_to_ko_model_name)\

        for i in range(len(self.data)):
            original_frame = self.data.iloc[i]
            original_id = original_frame["ID"].split("_")[-1]
            original_text = original_frame["text"]
            original_target = original_frame["target"]
            
            aug_id = "augmented-v1_train_"+original_id
            aug_text_en = self.translate(original_text, ko_to_en_model, ko_to_en_tokenizer)
            aug_text_ko = self.translate(aug_text_en, en_to_ko_model, en_to_ko_tokenizer)
            aug_label = original_target

            ids.append(aug_id)
            aug_t.append(aug_text_ko)
            targets.append(aug_label)

        back_data = pd.DataFrame()
        back_data["ID"] = ids
        back_data["text"] = aug_t
        back_data["target"] = targets

        print("Complete Backtranslation")

        return back_data