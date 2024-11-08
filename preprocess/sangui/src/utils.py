import re
import pandas as pd
import torch
from torch.utils.data import Dataset

korean = r"ㄱ-ㅎㅏ-ㅣ가-힣"
english = r"a-zA-Z"
number = r"0-9"
space = r"\s"
punctuation = r".,"
asciicode = r"\x00-\x7F"
flags = re.UNICODE

class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }

    def __len__(self):
        return len(self.labels)
    
    def to(self, device):
        for i in range(len(self)):
            self.inputs[i]['input_ids'] = self.inputs[i]['input_ids'].to(device)
            self.inputs[i]['attention_mask'] = self.inputs[i]['attention_mask'].to(device)
            self.labels[i] = self.labels[i].to(device)
    
class MyBERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label).view(1,-1))

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'],
            'attention_mask': self.inputs[idx]['attention_mask'],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)
    
    def to(self, device):
        for i in range(len(self)):
            self.inputs[i]['input_ids'] = self.inputs[i]['input_ids'].to(device)
            self.inputs[i]['attention_mask'] = self.inputs[i]['attention_mask'].to(device)
            self.labels[i] = self.labels[i].to(device)


def normalize(text):
    return text

def tokenize(text, tokenize_fn):
    text = map(tokenize_fn, text)
    return text

def remove_characters(text, pattern, flags=0):
    text = re.sub(pattern, "", text, flags)
    return text

def remove_korean(text):
    text = re.sub(f"[{korean}]", "", text, flags=flags)
    return text

def remove_korean_with_space(text):
    text = re.sub(f"[{korean+space}]", "", text, flags=flags)
    return text

def remove_not_korean(text):
    text = re.sub(f"[^{korean}]", "", text, flags=flags)
    return text

def remove_not_korean_without_space(text):
    text = re.sub(f"[^{korean+space}]", "", text, flags=flags)
    return text

def remove_english(text):
    text = re.sub(f"[{english}]", "", text, flags=flags)
    return text

def remove_english_with_space(text):
    text = re.sub(f"[{english+space}]", "", text, flags=flags)
    return text

def remove_not_english(text):
    text = re.sub(f"[^{english}]", "", text, flags=flags)
    return text

def remove_not_english_without_space(text):
    text = re.sub(f"[^{english+space}]", "", text, flags=flags)
    return text 

def remove_asciicode(text):
    text = re.sub(f"[{asciicode}]", "", text, flags=flags)
    return text

def remove_not_asciicode(text):
    text = re.sub(f"[^{asciicode}]", "", text, flags=flags)
    return text

def remove_not_asciicode_without_space(text):
    text = re.sub(f"[^{asciicode+space}]", "", text, flags=flags)
    return text

def remove_not_asciicode_with_space(text):
    text = re.sub(f"[^{asciicode}]", "", text, flags=flags)
    text = re.sub(f"[{space}]", "", text, flags=flags)
    return text

def remove_not_asciicode_with_space_period(text):
    text = re.sub(f"[^{asciicode}]", "", text, flags=flags)
    text = re.sub(f"[{space}]", "", text, flags=flags)
    text = re.sub(f"[.]", "", text, flags=flags)
    return text

def replace(pattern, replaced, text):
    text = re.sub(pattern, replaced, text)
    return text

def replace_asciicode():
    pass

def remove_special_characters(text):
    re.sub("", "", text)
    return text

def remove_not_special_characters(text):
    return text

def remove_repeated_characters(text):
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Regex pattern to match a word that repeats 3 or more times
    pattern = r'\b(\w+)(?:\s+\1){1,}\b'
    # Replace matched patterns with just one instance of the word
    text = re.sub(pattern, r'\1', text)
    return text

def remove_extension(text):
    ext_idx = text.rfind(".")
    return text[:ext_idx]

def calculate_ascii_ratio(text):
    origin_length = len(text)
    asciicode_with_space_length = len(remove_not_asciicode_with_space(text))
    ratio = asciicode_with_space_length/origin_length*100
    return ratio