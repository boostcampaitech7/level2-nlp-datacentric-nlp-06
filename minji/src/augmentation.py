import os
import sys
import time
import logging
import deepl
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from noise_score import noise_score_spacy


class Augmentation:
    def __init__(self, auth_key):
        self.translator = deepl.Translator(auth_key)

    def translate_with_retry(self, text, source_lang, target_lang, max_retries=5):
        for attempt in range(max_retries):
            try:
                return self.translator.translate_text(text, source_lang=source_lang, target_lang=target_lang).text
            except deepl.DeepLException as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Translation error occurred: {e}. Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    logging.error(f"Max retries reached. Translation failed for text: {text}")
                    return None

    def back_translation(self, data, save_path, trip_lang="JA"):
        # 폴더가 없으면 생성
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 30 <= noise_score < 70 데이터만 필터링
        data['noise_score'] = data['text'].apply(noise_score_spacy)
        data_denoise = data[(data.noise_score >= 30) & (data.noise_score < 70)].reset_index(drop=True)

        data_denoise['trans'] = ""
        data_denoise['trans_final'] = ""

        for idx, row in data_denoise.iterrows():
            text = row['text']

            trans = self.translate_with_retry(text, source_lang="KO", target_lang=trip_lang)
            if trans is not None:
                data_denoise.at[idx, 'trans'] = trans

                trans_final = self.translate_with_retry(trans, source_lang=trip_lang, target_lang="KO")
                if trans_final is not None:
                    data_denoise.at[idx, 'trans_final'] = trans_final
                    logging.info(f"{idx}: {text} -> {trans_final}")
                else:
                    logging.warning(f"Failed to translate from {trip_lang} to Korean for index {idx}")
            else:
                logging.warning(f"Failed to translate from Korean to {trip_lang} for index {idx}")

        # 최종 결과 필터링 및 저장
        data_denoise['text'] = data_denoise['trans_final']
        data_denoise = data_denoise.dropna(subset=['text'])
        data_denoise = data_denoise[['ID', 'text', 'target']]
        
        output_path = os.path.join(save_path, "augmentation.csv")
        data_denoise.to_csv(output_path, index=False)
        logging.info(f"Augmented data saved to {output_path}")
