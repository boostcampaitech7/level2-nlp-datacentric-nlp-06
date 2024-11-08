# Data preprocessing & Augmentation

## Component

### Relabeling + Text filtering
* SBERT 기반으로 Noise가 없는 데이터에 한해 relabeling 수행 (Pollution 30 up)
* Gemma2-9B 기반으로 Prompting하여 ReLabeling한 데이터에서의 data Filtering 수행
* (opt) MLM모델인 "BART" 기반 토큰 생성 방식으로 Data Filtering 수행

### Backtranslation
* nllb Huggingface 모델 기반으로 BackTranslation 수행
* Relabeling + Text filtering한 데이터를 기반으로 데이터 증강 시도
* Korean → English → Korean 수행

### Gemma Prompting Augmentation
* Gemma2 한국어 모델을 기반으로 Synthetic Data 생성
* Few-shot으로 각 label마다 40개씩 주어 100개의 문장을 생성
* 최대한 중복을 피하고자 프롬프팅에도 명시하고 temperature 값 0.9를 주었음


## How to start
```shell
$ python prep_main.py --data_path {train_data_path} --filter {option}
```
* `data_path` : `train.csv` 데이터가 존재하는 폴더 위치를 지정
* `filter` : `BART`, `gemma` 옵션으로 선택가능
    * `BART` : BART 기반 `<mask>` 토큰 추정 방식
    * `gemma` : 생성형 모델 gemma 기반 토큰 생성 방식

* 최종적으로 `{data_path}/train_prep_aug.csv`로, 총 1504(filter)+300(bt)+700(gemma) = 2504개의 데이터가 저장됨

### 결과

| Method                       | F1(val) | ACC(test)  | F1(test) | Version | 총 데이터 수 |
|--------------------------------------|-----------|---------|----------|---------|--------------|
| Relabeling + Text Filtering + BackTranslation(300) + Synthetic Data (700) | 0.85      | 0.6813  | 0.6634   | v0.2    | 2704 
