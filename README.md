### Data-Centric Baseline Code
Data-Centrin 프로젝트를 위한 베이스라인 코드를 재작성했습니다.  
코드가 복잡하지 않아 별도의 폴더들로 구분하지 않았습니다.  

### How To
requirements는 기존 코드의 requirements와 동일합니다.  
```bash
python main.py --data ../data_path --model ./model_path --mode train
```
`--data`  
데이터가 있는 폴더의 경로, default: ../../data  

`--model`  
모델을 저장할 폴더 이름, default: ../model  

`--mode`  
**train**, **test** 중 선택, default: train  
**train:** 학습 후 train test split 후 얻은 validation datset으로 추론 수행  
**test:** 위 --model 옵션에서 입력한 경로의 모델을 불러와 추론 수행. 즉, 반드시 train한 후 수행해야 함  

### 간단한 설명
- train test split 비율을 변경하고 싶다면 [trainer.py](./trainer.py)에 이동해서 `train_test_split` 함수를 사용하는 곳에서 변경하면 됩니다(default 0.3, 게시판 권장 0.2).  
- 학습 중에는 checkpoint가 두 개 저장되고, 학습이 끝난 후에는 최종 모델만 저장됩니다.  
- 서로 다른 데이터셋들로 테스트 할 상황을 가정하여 validation split과 test 데이터셋으로 추론하는 경우, 해당 데이터셋 폴더 내부에 결과가 저장됩니다.  
- 학습 과정에 validation dataset을 사용한 f1 점수와 loss를 확인할 수 있습니다.  