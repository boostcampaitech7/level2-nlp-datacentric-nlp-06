# Data-Centric 주제 분류 프로젝트
모델 구조를 변경하지 않고, Data-Centric 관점으로 데이터 품질을 개선하여 텍스트 주제 분류 모델 성능을 향상시키는 프로젝트입니다. 학습 데이터의 노이즈 및 레이블 오류를 정제하고, 부족한 데이터를 합성하여 데이터 균형을 개선하는 것이 목표입니다.

## 주요 접근
#### 1. 데이터 정제
- **텍스트 노이즈 제거**: spacy 라이브러리를 사용하여 노이즈를 계산하고, Llama 모델을 활용하여 텍스트를 denoising.
- **레이블 재부여**: Llama 모델을 이용해 레이블 오류가 있는 데이터를 재라벨링하여 레이블 품질 개선.
#### 2. 합성 데이터 생성
- 레이블 별 데이터 수를 100개 이상 생성하여 데이터 추가.
- 모델이 특정 레이블(0, 4, 5)을 잘 탐지하지 못하는 경우에 대해 few-shot을 활용한 데이터 증강
    - label 0: 문화 생활 관련 text 생성 ([few-shot](./prompt/synthetic_fewshot_0.json))
    - label 4: 과학 기술 관련 text 생성 ([few-shot](./prompt/synthetic_fewshot_4.json))
    - label 5: IT 기업의 경제 소식 관련 text 생성 ([few-shot](./prompt/synthetic_fewshot_5.json))

## 결과
- **평가기준**: Macro F1-score
- **모델 성능**: 0.7759 (baseline 0.5668 대비 0.2091 향상)

## 실행방법 (`minji` 폴더 기준 )
1. Requirements 설치
```bash
pip install -r requirements.txt
python -m spacy download ko_core_news_sm
```

2. `main.py` 실행 - 최종 성능을 낸 데이터셋 생성
```bash
python main.py
```

## 모듈 설명 (`minji/src` 폴더)
- `noise_score.py`: 데이터 노이즈 계산
    - `noise_score_spacy()`: 텍스트 노이즈 점수 계산
- `llama.py`: `meta-llama/Llama-3.1-8B-Instruct` 모델 활용 모듈
    - `inference()`: 프롬프트 및 few-shot 설정을 통해 텍스트 생성
- `clean.py`: 데이터 전처리 모듈
    - `clean_characters()`: 텍스트 정제 (`Llama-3.1-8B` 모델)
    - `clean_labels_cleanlab()`: 레이블 정제 (`cleanlab` 라이브러리)
    - `clean_labels_llama()`: 레이블 재부여 (`Llama-3.1-8B` 모델)
    - `delete_similar_text()`: 유사한 텍스트 제거 (`SBERT` 모델 및 `KMeans` 알고리즘 활용)
    - `plot_noise_scores()`: 노이즈 점수 시각화 (히스토그램, 박스플롯)
- `synthetic.py`: 합성 데이터 생성 모듈 (`Llama-3.1-8B` 모델)
    - `generate_all_targets()`: 모든 레이블에 대해 일괄 텍스트 생성
    - `generate_targetwise()`: 특정 레이블에 대한 텍스트 생성
- `augmentation.py`: Back Translation을 활용한 데이터 증강
    - `back_translation()`: DeepL API로 번역 후 재번역 (성능 향상되지 않아 최종 모델에는 제외)

## 기타 파일
- `main.py`: 최종 데이터셋 생성 스크립트
- `generate_validset.py`: 학습 데이터 오염 문제로 인해, 평가용 validation set을 별도로 구성하여 초기 성능 평가를 보다 명확하게 하기 위한 파일

## 데이터셋 버전 관리
- v0.0.2: validation과 train set 구분 (optional)
- v0.0.3: 텍스트 denoising 적용
- v0.0.4: Llama를 활용한 레이블 정제
- v0.0.5: Cleanlab으로 품질 낮은 레이블 삭제
- v1.0.1: 7개 레이블에 대해 각 100개의 합성 데이터 생성
- v1.0.2: 유사한 텍스트 삭제로 데이터 정제
- v1.0.3: v1.0.2에서 발생한 레이블 오류 데이터 제거
- v1.1.1: 타겟별 특정 레이블(0, 4, 5)에 대해 200개씩 데이터 증강
- v1.2.1: v1.0.3과 v1.1.1의 데이터를 병합
