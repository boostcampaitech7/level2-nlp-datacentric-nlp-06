import pandas as pd
from googletrans import Translator
import deepl
import difflib
import transformers
import torch
from huggingface_hub import login
import tqdm
import argparse

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Translation:
    def __init__(self, api_key, dataset_path) -> None:
        self.api_key = api_key
        self.google_translator = Translator()
        self.deepl_translator = deepl.Translator(api_key)
        self.dataset = pd.read_csv(dataset_path)

    def eval_metrics(self, original, sentence):
        # SequenceMatcher를 사용한 문장 유사도 평가 페트릭
        answer_bytes = bytes(original, 'utf-8')
        input_bytes = bytes(sentence, 'utf-8')
        answer_bytes_list = list(answer_bytes)
        input_bytes_list = list(input_bytes)

        sm = difflib.SequenceMatcher(None, answer_bytes_list, input_bytes_list)
        similar = sm.ratio()

        return similar
    
    def get_translated_df(self, display_result=False):
        id, text, target, google_evals, deepl_evals = [], [], [], [], []

        for i in tqdm(range(len(self.dataset))):
            avg = 0
            data = self.dataset.iloc[i]["text"]

            google_ko_en = self.google_translator.translate(data, dest="en").text
            if len(google_ko_en) > 0:
                google_en_ko = self.google_translator.translate(google_ko_en, dest="ko").text

            if len(google_en_ko) > 0:
                google_seque_eval = self.eval_metrics(data, google_en_ko)

            deepl_ko_en = self.deepl_translator.translate_text(data, target_lang="EN-US").text
            if len(deepl_ko_en) > 0:
                deepl_en_ko = self.deepl_translator.translate_text(deepl_ko_en, target_lang="KO").text

            if len(deepl_en_ko) > 0:
                deepl_seque_eval = self.eval_metrics(data, deepl_en_ko)

            if google_seque_eval > 0 and deepl_seque_eval > 0:
                avg = (google_seque_eval + deepl_seque_eval) / 2

            if avg >= 0.6 and (max(google_seque_eval, deepl_seque_eval) - min(google_seque_eval, deepl_seque_eval) < 0.4):
                if display_result:
                    print(f"original : {data}")
                    print(f"google_en->ko : {google_en_ko}")
                    print(f"deepl_en->ko : {deepl_en_ko}")
                    print(f"average : {avg :.2f}")
                    print(f"google_seque_eval : {google_seque_eval :.2f}")
                    print(f"deepl_seque_eval : {deepl_seque_eval :.2f}")
                    print("=" * 30)

                id.append(self.dataset.iloc[i]["ID"])
                text.append(data)
                target.append(self.dataset.iloc[i]["target"])
                google_evals.append(google_seque_eval)
                deepl_evals.append(deepl_seque_eval)

        new_df = pd.DataFrame({
            "ID": id,
            "text": text,
            "target": target,
            "google_seque_eval": google_evals,
            "deepl_seque_eval": deepl_evals
        })
        
        print(f"총 데이터 개수 : {len(new_df)}")
        
        return new_df

    
class Llama:
    def __init__(self, api_key, path, save_path, model_id) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.dataset = pd.read_csv(path)
        self.save_path = save_path
        self.pipeline = self.pipeline_init()

        login(token=self.api_key)

    def pipeline_init(self):
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        pipeline.model.eval()

        return pipeline
    
    def llama_message(self, messages):
        message_output = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.pipeline(
            message_output,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        return outputs[0]["generated_text"][len(message_output):]
        
    def get_lebel(self, prompt):
        sam_df = self.dataset.sample(n=20).reset_index(drop=True)
        data = ", ".join(sam_df['text'] + ":" + sam_df['target'].astype(str))

        messages = [
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"{data}"}
        ]

        result = self.llama_message(messages)

        return result
    
    def relabeling(self, prompt):
        keywords = []
        for i in tqdm(range(len(self.dataset))):
            data = self.dataset.iloc[i]["text"]

            messages = [{"role": "system", "content": f"{prompt}"}] + \
                        [{"role": "user", "content": f"{data}"}]

            key = self.llama_message(messages)

            keywords.append(int(key))
        self.dataset["new_target"] = keywords
        self.remove_nosie(40)

    def calculate_ascii_noise(self, text):
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
    
    def remove_nosie(self, threshold):
        text_datas = self.dataset["text"]
        asc_counts = []; non_asc_counts = []; percentages = []
        # ASCII 문자 개수, 비-ASCII 문자 개수, ASCII 비율 계산
        for text_with_noise in text_datas:
            ascii_count, non_ascii_count, ascii_percentage = self.calculate_ascii_noise(text_with_noise)
            ascii_percentage = round(ascii_percentage,2)
            asc_counts.append(ascii_count)
            non_asc_counts.append(non_ascii_count)
            percentages.append(ascii_percentage)

        print(len(asc_counts), len(non_asc_counts), len(percentages))
        self.dataset["ratio"] = percentages
        self.dataset = self.dataset[self.dataset['ratio'] < threshold]
        self.dataset['target'] = self.dataset['new_target']
        self.dataset = self.dataset.drop(columns=['new_target'])

        self.dataset.to_csv(self.save_path)

class Mymain():
    def __init__(self, data_path, model_id):
        self.data_path = data_path
        self.model_id = model_id

    def main(self):
        deepl_api_key = "your_deepl_key"

        translation = Translation(deepl_api_key, self.data_path)
        translation_df = translation.get_translated_df()

        llama_prompt_1 = '''
            당신은 기사 제목을 보고 어떤 분야의 기사인지 맞추는 전문가입니다.
            기사 제목이 주어지면 기사 제목을 가장 잘 표현하는 포괄적인 분야를 한 단어로 출력하세요.
            target 값 별로 분야를 지정해주세요.
            출력 형식은 0:분야1, 1:분야2, ... , 6:분야3 과 같은 형식으로 출력하세요.
            입력은 기사제목:분야, 기사제목:분야 와 같은 형식으로 입력됩니다.
        '''

        hugginface_api_key = "your_huggingface_api_key"

        llama = Llama(hugginface_api_key, self.data_path, self.model_id)
        keywords_by_llama = llama.get_lebel(llama_prompt_1)

        llama_prompt_2 = f'''
            당신은 기사 제목을 보고 어떤 분야의 기사인지 맞추는 전문가입니다.
            기사의 분야에 대한 정수 값은 {keywords_by_llama}와 같습니다.
            기사를 보고 해당하는 분야의 정수 값을 리턴해주세요.
            결과 값에 숫자만 있게 해주세요. 글자 하나 없이 오로지 숫자만
        '''
        llama.relabeling(llama_prompt_2)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../dataset", help="path where data csv is stored")
    parser.add_argument("--model", type=str, default="MLP-KTLim/llama-3-Korean-Bllossom-8B", help="model name in huggingface")
    args = parser.parse_args()
    mymain = Mymain(args.data, args.model)
    mymain.main()