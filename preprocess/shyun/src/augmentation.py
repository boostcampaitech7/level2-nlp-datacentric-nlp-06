import random
import os, re
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Augmentation:    
    def back_translation(self, train_dataset, path):
        translator_1 = pipeline('translation', model='NHNDQ/nllb-finetuned-en2ko', device=0, src_lang='kor_Hang', tgt_lang='eng_Latn', max_length=512)
        translator_2 = pipeline('translation', model='NHNDQ/nllb-finetuned-en2ko', device=0, src_lang='eng_Latn', tgt_lang='kor_Hang', max_length=512)

        bt = pd.DataFrame(columns=['ID', 'text', 'target', 'polluted_lv'])
        for i in tqdm(range(7), desc='rtt', total=7, position=0):
            target = train_dataset[(train_dataset['target']==i) & (train_dataset['polluted_lv']<=0.10)]['text'].values

            bt_tmp = pd.DataFrame(columns=['ID', 'text', 'target', 'polluted_lv'])
            for j, text in tqdm(enumerate(target), desc=f'{i}', total=len(target), position=1, leave=False):
                output = translator_1(text, max_length=512)
                output = translator_2(output[0]['translation_text'], max_length=512)

                bt_tmp.loc[j] = [f'rtt-{i}-{j}', output[0]['translation_text'], i, 0]

            bt = pd.concat([bt, bt_tmp])

        bt.to_csv(os.path.join(path, 'back_translation.csv'), index=False)
    
    def eda_sr(self, model_path, train_dataset, path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

        vocab_embeddings_dict = []
        for vocab, _ in tqdm(tokenizer.vocab.items(), desc='embedding', total=len(tokenizer.vocab)):
            embedding = model.encode(vocab)
            vocab_embeddings_dict.append({'vocab': vocab, 'embedding': embedding})

        vocab_embeddings = [v['embedding'] for v in vocab_embeddings_dict]
        vocab_embeddings = np.array(vocab_embeddings)

        synonym = pd.DataFrame(columns=['ID', 'text', 'target', 'polluted_lv'])
        for i, data in tqdm(train_dataset.iterrows(), desc='replacing', total=len(train_dataset)):
            parsed = re.compile('/ ^[ㄱ-ㅎㅏ-ㅣ가-힣]*$/').sub('',data['text'])
            words = parsed.split(' ')

            if len(words) >= 3: # 한 개일 때가 제일 나았음
                target_idx = random.sample(range(0, len(words)), 3)
            elif len(words) >= 2:
                target_idx = random.sample(range(0, len(words)), 2)
            else:
                target_idx = random.sample(range(0, len(words)), 1)

            for t_idx in target_idx:
                target_embedding = np.array([model.encode(words[t_idx])])
                
                cos_sim = cosine_similarity(target_embedding, vocab_embeddings)
                sorted_indices = np.argsort(cos_sim, axis=1)[:, ::-1]
                idx = sorted_indices[0][10]

                words[t_idx] = vocab_embeddings_dict[idx]['vocab']
            
            target = data['target']
            synonym.loc[i] = [f'synonym-{target}-{i}', ' '.join(words), target, 0]

        synonym.to_csv(os.path.join(path, 'synonym_replaced.csv'), index=False)