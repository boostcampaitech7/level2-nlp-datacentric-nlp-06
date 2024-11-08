from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class SBERT:
    def __init__(self, data):
        self.data = data
        self.model_path = "sinjy1203/ko-sbert-navernews"
        self.model = SentenceTransformer(self.model_path)

    def get_embeddings(self, texts):
        return self.model.encode(texts)

    def clustering(self, k):
        embeddings = self.get_embeddings(self.data["text"])
        kmeans = KMeans(n_clusters=7, random_state=456).fit(embeddings)
        return kmeans
    
    def mapping(self, kmeans):

        # random_state=456일 때의 맵핑 정보
        """
        실제로는 clustering한 label로 써야하나, 정답이 있기 때문에 현재는 수동적으로 맵핑을 진행
        """
        def transform_values(input_list):
            # 매핑 딕셔너리 정의
            mapping_num = {
                4: 0,
                2: 1,
                1: 2,
                6: 3,
                5: 4,
                0: 5,
                3: 6
            }
            
            # 입력 리스트의 각 요소를 매핑 딕셔너리를 사용해 변환
            transformed_list = [mapping_num.get(value, value) for value in input_list]
            return transformed_list
        
        relabeled_mapping_list = transform_values(kmeans.labels_)
        self.data["target"] = relabeled_mapping_list
        return self.data

