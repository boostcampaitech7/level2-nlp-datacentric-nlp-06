import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "beomi/Llama-3-Open-Ko-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")


words = ["대통령", "게시판", "출시", "이란", "삼성", "한국", "감독", "개발", "네이버", 
         "분기", "억원", "트럼프", "만에", "개최", "개막", "갤럭시", "신간", "전국", "추진", "증가", "류현진", "정부", "내년", 
         "경기", "연속", "공개", "프로농구", "코스피", "올해", "김정은", "애플", "스마트폰", "여행", "서비스", "민주", "그래픽", "최대",
           "오늘", "지원", "아이폰", "성공", "아시안게임", "다시", "국내", "종합", "홍콩", "세계", "선정", "대표", "하락"]


prompt = f"""다음 단어 목록을 사용하여 7개의 뉴스 카테고리로 분류해주세요:

{', '.join(words)}

각 카테고리를 다음 형식으로 제시해주세요:

Keyword: [카테고리 이름]
Potential Categories: [관련 단어들을 쉼표로 구분하여 나열]

예시:
Keyword: 정치/외교
Potential Categories: 대통령, 트럼프, 김정은, 정부, 민주, 이란, 홍콩

위 예시와 같은 형식으로 7개의 카테고리를 모두 작성해주세요. 각 카테고리는 고유해야 하며, 단어들이 중복되지 않도록 해주세요."""



inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.5)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)