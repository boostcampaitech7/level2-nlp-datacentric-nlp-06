import os
from tqdm import tqdm

import torch
import numpy as np
from cleanlab.filter import find_label_issues
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CleanLab():
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint")]
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1])) if checkpoints else None
        checkpoint_path = os.path.join(model_path, latest_checkpoint)

        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=7).to(DEVICE)
        self.model.eval()
    
    def clean_labels(self, train_dataset, path):
        preds = []
        for _, data in tqdm(train_dataset.iterrows(), desc='logits', total=len(train_dataset)):
            tokenized = self.tokenizer(data['text'], padding='max_length', max_length=50, truncation=True, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = self.model(**tokenized).logits.cpu().numpy()
                softmax = [np.exp(x)/np.sum(np.exp(logits)) for x in logits]
                preds.append({'ID': data['ID'], 'softmax': softmax})

        pred_probs = [np.array(p['softmax'][0]) for p in preds]
        pred_probs = np.array(pred_probs)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        ordered_label_issues = find_label_issues(
            labels=train_dataset['target'],
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence'
        )

        print(f"total {len(ordered_label_issues)}")
        for issue in ordered_label_issues:
            pred = np.argmax(pred_probs[issue])

            print('input text:', train_dataset.iloc[issue]['text'])
            print('label:', train_dataset.iloc[issue]['target'])
            print('pred:', pred)
            print('------------------------------------------------------')

            train_dataset.loc[train_dataset['text']==train_dataset.iloc[issue]['text'], 'target'] = pred

        train_dataset.to_csv(os.path.join(path, 'label_cleaned.csv'), index=False)