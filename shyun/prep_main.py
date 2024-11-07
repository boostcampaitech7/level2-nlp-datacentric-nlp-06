import os, json
import argparse
import pandas as pd

from llama import Llama
from clean import Clean
from augmentation import Augmentation

def main():
    train_dataset = pd.read_csv('../../v1.3.0/train.csv')
    
    with open('key_maps.json', 'r', encoding='utf-8') as f:
        keys = json.load(f)
    keys = list(keys.values())
    
    # lama = Llama()
    # lama.extract_label(train_dataset)
    # lama.clean_text(list(keys.values()), train_dataset, p=0.3, path='./test')
    # lama.clean_label(list(keys.values()), train_dataset, p=0.2, path='./test')
    # lama.generate_new(list(keys.values()), train_dataset, num=10, path='./test')
    # lama.regenerate(list(keys.values()), train_dataset, num=10, path='./test')

    # cl = Clean()
    # valid_predictions = pd.read_csv('../../v1.3.0/valid_output.csv)
    # cl.check_f1(train_dataset, valid_predictions, path='./test)
    # cl.clean_labels('../../model-v1.3.6', train_dataset, path='./test')
    # cl.clean_characters(train_dataset, path='./test')

    # aug = Augmentation()
    # aug.back_translation(train_dataset, path='./test')
    # aug.eda_sr(model_path='../../model-v1.3.6', train_dataset=train_dataset, path='./test')

if __name__ == "__main__":
    main()