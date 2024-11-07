# !huggingface-cli login 후 진행
import pandas as pd
from src.clean import Clean
from src.synthetic import SyntheticData

clean = Clean()
system_prompt_path = "./prompt/synthetic_prompt.txt"
syn = SyntheticData(system_prompt_path)



## 데이터 전처리 ------------------------------------------------------

### v0.0.3. text denoising
data = pd.read_csv("../../data/train.csv")
system_prompt_path = "./prompt/denoise_prompt.txt"
fewshot_path = "./prompt/denoise_fewshot.json"
save_path = "../../datasets/v0.0.3/"
clean.clean_characters(data, system_prompt_path, fewshot_path, save_path)

### v0.0.4. relabeling
# python ../main.py --data ../../datasets/v0.0.3 --model ../model --mode train 후 진행
model_path = "../model"
data = pd.read_csv("../../datasets/v0.0.3/train.csv")
system_prompt_path = "./prompt/label_prompt.txt"
fewshot_path = "./prompt/label_fewshot.json"
save_path = "../../datasets/v0.0.4/"
clean.clean_labels_llama(model_path, data, system_prompt_path, fewshot_path, save_path)

### v0.0.5 라벨 오류 제거
# python ../main.py --data ../../datasets/v0.0.4 --model ../model --mode train 후 진행
model_path = "../model"
data = pd.read_csv("../../datasets/v0.0.4/train.csv")
save_path = "../../datasets/v0.0.5/"
clean.clean_labels_cleanlab(model_path, data, save_path)


## 데이터 증강 ------------------------------------------------------

# v1.0.1 target별 100개 데이터 증강
save_path = "../../datasets/v1.0.1/"
fewshot_data = pd.read_csv("../../datasets/v0.0.5/train.csv")
n_random_fewshots = 30
num = 100
syn.generate_all_targets(save_path, fewshot_data, n_random_fewshots, num)

# v1.0.2 target별 유사한 데이터 삭제
train_data = pd.read_csv("../../datasets/v0.0.5/train.csv")
synthetic_data = pd.read_csv("../../datasets/v1.0.1/synthetic.csv")
save_path = "../../datasets/v1.0.2/"
similarity_threshold = 0.85
clean.delete_similar_text(train_data, synthetic_data, save_path, similarity_threshold, filename="synthetic.csv")

# v1.0.3 라벨 오류 제거
org_train = pd.read_csv("../../datasets/v0.0.5/train.csv")
synthetic = pd.read_csv("../../datasets/v1.0.2/synthetic.csv")
new_train = pd.concat([org_train, synthetic], axis=0)
new_train.to_csv("../../datasets/v1.0.2/train.csv", index=False)
# python ../main.py --data ../../datasets/v1.0.2 --model ../model --mode train  후 진행
model_path = "../model"
data = pd.read_csv("../../datasets/v1.0.2/train.csv")
save_path = "../../datasets/v1.0.3/"
clean.clean_labels_cleanlab(model_path, data, save_path)

# v1.1.1 합성데이터 생성
targets=[0,4,5]
fewshot_path="./prompt/synthetic_fewshot"
save_path = "../../datasets/v1.1.1/"
num = 200
syn.generate_targetwise(targets, fewshot_path, save_path, num)

# v1.2.1 - v1.0.3과 v.1.1.1 데이터 합치기
data_103 = pd.read_csv("../../datasets/v1.0.3/train.csv")
data_111 = pd.read_csv("../../datasets/v1.1.1/synthetic.csv")
data = pd.concat([data_103, data_111], axis=0)
data.to_csv("../../datasets/v1.2.1/train.csv", index=False)



# terminal: "huggingface-cli login {hf_api_key}" 입력한 뒤 다음 코드 실행 
import pandas as pd
from src.clean import Clean
from src.synthetic import SyntheticData

def main():
    # 클래스 초기화
    clean = Clean()
    syn = SyntheticData(system_prompt_path="./prompt/synthetic_prompt.txt")

    ## 데이터 전처리 ------------------------------------------------------

    # v0.0.3 - Text Denoising
    data = pd.read_csv("./data/train.csv")
    save_path_v003 = "./datasets/v0.0.3/"
    clean.clean_characters(
        data, 
        system_prompt_path="./prompt/denoise_prompt.txt", 
        fewshot_path="./prompt/denoise_fewshot.json", 
        save_path=save_path_v003
    )

    # 모델 학습 (v0.0.3 데이터 기반)
    print("Please run the following command to train the model:")
    print("python ../main.py --data ./datasets/v0.0.3 --model ../model --mode train")
    input("Press Enter after model training is complete...")

    # v0.0.4 - Relabeling
    data = pd.read_csv(f"{save_path_v003}train.csv")
    save_path_v004 = "./datasets/v0.0.4/"
    clean.clean_labels_llama(
        model_path="../model",
        data=data,
        system_prompt_path="./prompt/label_prompt.txt",
        fewshot_path="./prompt/label_fewshot.json",
        save_path=save_path_v004
    )

    # 모델 학습 (v0.0.4 데이터 기반)
    print("Please run the following command to train the model:")
    print("python ../main.py --data ./datasets/v0.0.4 --model ../model --mode train")
    input("Press Enter after model training is complete...")

    # v0.0.5 - 라벨 오류 제거
    data = pd.read_csv(f"{save_path_v004}train.csv")
    save_path_v005 = "./datasets/v0.0.5/"
    clean.clean_labels_cleanlab(
        model_path="../model",
        data=data,
        save_path=save_path_v005
    )

    ## 데이터 증강 ------------------------------------------------------

    # v1.0.1 - Target별 데이터 증강
    fewshot_data = pd.read_csv(f"{save_path_v005}train.csv")
    save_path_v101 = "./datasets/v1.0.1/"
    syn.generate_all_targets(
        save_path=save_path_v101, 
        fewshot_data=fewshot_data, 
        n_random_fewshots=30, 
        num=100
    )

    # v1.0.2 - Target별 유사한 데이터 삭제
    train_data = pd.read_csv(f"{save_path_v005}train.csv")
    synthetic_data = pd.read_csv(f"{save_path_v101}synthetic.csv")
    save_path_v102 = "./datasets/v1.0.2/"
    clean.delete_similar_text(
        train_data=train_data,
        synthetic_data=synthetic_data,
        save_path=save_path_v102,
        similarity_threshold=0.85,
        filename="synthetic.csv"
    )

    # v1.0.3 - 라벨 오류 제거 후 합치기
    org_train = pd.read_csv(f"{save_path_v005}train.csv")
    synthetic = pd.read_csv(f"{save_path_v102}synthetic.csv")
    new_train = pd.concat([org_train, synthetic], axis=0)
    new_train.to_csv(f"{save_path_v102}train.csv", index=False)

    # 모델 학습 (v1.0.2 데이터 기반)
    print("Please run the following command to train the model:")
    print("python ../main.py --data ./datasets/v1.0.2 --model ../model --mode train")
    input("Press Enter after model training is complete...")

    save_path_v103 = "./datasets/v1.0.3/"
    clean.clean_labels_cleanlab(
        model_path="../model",
        data=new_train,
        save_path=save_path_v103
    )

    # v1.1.1 - 추가 합성 데이터 생성
    targets = [0, 4, 5]
    save_path_v111 = "./datasets/v1.1.1/"
    syn.generate_targetwise(
        targets=targets, 
        fewshot_path="./prompt/synthetic_fewshot", 
        save_path=save_path_v111, 
        num=10
    )

    # v1.2.1 - 최종 데이터 병합
    data_103 = pd.read_csv(f"{save_path_v103}train.csv")
    data_111 = pd.read_csv(f"{save_path_v111}synthetic.csv")
    final_data = pd.concat([data_103, data_111], axis=0)
    final_data.to_csv("../../datasets/v1.2.1/train.csv", index=False)

if __name__ == "__main__":
    main()
