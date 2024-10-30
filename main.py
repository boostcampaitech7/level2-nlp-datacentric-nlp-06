import os
import torch
import random
import argparse
import numpy as np

from trainer import MyTrainer
import transformers

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data', help='path where data csv is stored')
    parser.add_argument('--model', type=str, default='./model', help='path for saving model during training and loading during testing')
    parser.add_argument('--mode', type=str, default='train', help='whether to train, valid or test')
    args = parser.parse_args()

    trainer = MyTrainer(args.data, args.model)

    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        trainer.test()
    else:
        print("Invalid task! Please choose either 'train' or 'test'")
