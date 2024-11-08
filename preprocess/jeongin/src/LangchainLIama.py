from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import Counter
import torch
import pandas as pd

#pip install transformers langchain torch pandas

import os
os.environ["HUGGINGFACE_TOKEN"] = ""

# Llama 모델 로드 (예: meta-llama/Llama-2-7b-chat-hf)잘 나오지 않아 라마 모데 교체 beomi/Llama-3-Open-Ko-8B
model_name = "beomi/Llama-3-Open-Ko-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# HuggingFace 파이프라인 설정
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.5,
    top_p=0.95,
    repetition_penalty=1.15
)

# LangChain의 HuggingFacePipeline 생성
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 프롬프트 템플릿 설정
prompt = PromptTemplate(
    input_variables=["text"],
    template="""다음은 뉴스 기사의 일부 단어들입니다:

{text}

위 단어들과 가장 관련 있는 뉴스 도메인을 한 단어로 정의해 주세요. 예시: 정치, 경제, 사회, 문화, 국제, 과학, 스포츠, 기술.
다른 설명이나 추가 정보 없이 오직 도메인 단어 하나만 답하세요.

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
for text, category in zip(texts[:10], categories[:10]):  # 처음 10개만 출력
    print(f"Text: {text}\nCategory: {category}\n")

# 카테고리 분포 확인
category_counts = Counter(categories)
print("Category distribution:")
for category, count in category_counts.most_common(7):  # 상위 7개 카테고리만 출력
    print(f"{category}: {count}")

# 상위 7개 카테고리 추출
top_7_categories = [category for category, _ in category_counts.most_common(7)]
print("\nTop 7 categories:", top_7_categories)