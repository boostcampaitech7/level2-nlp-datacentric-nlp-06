# # 라이브러리 설치
# !pip install langchain
# !pip install langchain_openai
# !pip install langchain_experimental
# !pip install openai
# !pip install wikipedia
# !pip install chromadb


from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import warnings
import os

os.environ["OPENAI_API_KEY"] = "" # OpenAI API key 설정


# 사용할 LLM 모델 설정
llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',  # 'gpt-4-mini' 대신 사용 가능한 모델명으로 변경
    temperature=0.5
)

# 프롬프트 템플릿 설정
prompt = PromptTemplate(
    input_variables=["text"],
    template="""다음은 뉴스 기사의 일부 단어들입니다:

{text}

위 단어들과 가장 관련 있는 뉴스 도메인을 한 단어로 정의해 주고 가장 관련 있는 카테고리 찾아주세요.
그 후 평균적으로 가장 많이 나온 카테고리 7가지 결과 값을 찾아주세요.  

도메인:"""
)

# LLMChain 생성
chain = LLMChain(llm=llm, prompt=prompt)

# 텍스트 리스트에서 카테고리 추출
texts = df_cleaned['cleaned_text'].tolist()
categories = []

for text in texts:
    response = chain.run(text)
    categories.append(response.strip())

# 결과 출력
for text, category in zip(texts, categories):
    print(f"Text: {text}\nCategory: {category}\n")

# 카테고리 분포 확인
from collections import Counter
category_counts = Counter(categories)
print("Category distribution:")
for category, count in category_counts.most_common():
    print(f"{category}: {count}")