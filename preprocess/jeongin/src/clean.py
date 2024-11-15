import re
from konlpy.tag import Okt
import pandas as pd

#pip install konlpy pandas


#특수 기호 제거 함수:
def remove_special_characters(text):
    return re.sub(r'[^ ㄱ-ㅣ가-힣]', '', text)


#형태소 분석 및 불용어 제거 함수
okt = Okt()
korean_stopwords = set(['은', '는', '이', '가', '을', '를', '의', '에', '에서', '로', '으로', '과', '와', '도', '만', '에게', '께', '한테', '보다', '라고', '이라고', '으로서', '같이', '처럼', '만큼'])



def tokenize_and_remove_stopwords(text):
    tokens = okt.pos(text, stem=True)
    return ' '.join([word for word, pos in tokens if pos in ['Noun', 'Verb', 'Adjective'] and word not in korean_stopwords])



#2글자 이상 단어 필터링 함수
def filter_tokens_by_length(text, min_length=2):
    return ' '.join([word for word in text.split() if len(word) >= min_length])



#전체 전처리 함수
def preprocess_text(text):
    text = remove_special_characters(text)
    text = tokenize_and_remove_stopwords(text)
    text = filter_tokens_by_length(text)  # 2글자 이상 단어 필터링 추가
    return text


#DataFrame에 전처리 적용
def preprocess_dataframe(df):
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df



# 데이터프레임 전처리 예시)
#df_cleaned = preprocess_dataframe(df_train_pc)
