### Text Denoising, Relabeling
아스키코드로 대체된 텍스트의 원문 복구
레이블이 잘못 부여된 데이터의 레이블 재부여

### Usage
## Requirements
```bash
pip install -r requirements.txt
```

## Command
```bash
cd ./src
python run.py
```

## Config
```
"pivot_name": "ascii_ratio" # prefix of augmented data file name
"ascii_ratio": "20" # split text noise data/label error data by ascii ratio


"data_folder": "../data/" # folder of original data
"preprocess_data_folder": "../data/preprocess/" # augmented data folder
"train_data_folder": "../data/preprocess/train/" # not implemented
"valid_data_folder": "../data/preprocess/valid/" # not implemented

"device": "cuda" # 'cpu' or 'cuda'


# MaskedLanguageModel Setting
"mlm":
  "model_name": ""klue/bert-base"
  "model_output_dir": "../model/mlm/"
  "batch_size": 8
  "train": True

# BackTranslationModel Setting
"btm":
  "model_name": "facebook/nllb-200-distilled-600M"
  "model_output_dir": "../model/btm/"
  "batch_size": 8
  "train": True

# ReLabelingModel Setting
"rlm":
  "model_name": "klue/bert-base"
  "model_output_dir": "../model/rlm/"
  "batch_size": 32
  "train": True

# CleanLabModel Setting
"clm":
  "model_name": '../model/rlm'
  "model_output_dir": "../model/clm/"
  "batch_size": 32
  "train": True

# BaseLineModel Setting
"blm":
  "model_name": "klue/bert-base"
  "model_output_dir": "../model/blm/"
  "batch_size": 32
  "train": True
```