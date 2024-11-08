import spacy
import string
import re

# 한국어 모델 로드
nlp = spacy.load("ko_core_news_sm")

def noise_score_spacy(text):
    doc = nlp(text)  # 입력된 텍스트 토큰화
    word_count = len(doc) # 토큰 수 count
    
    # 노이즈 관련 점수 계산
    special_char_count = sum(1 for char in text if char in string.punctuation) # 특수문자의 개수 세기: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    non_korean_char_count = len(re.findall(r'[a-zA-Z0-9]', text))  # 한글이 아닌 문자(알파벳, 숫자) 개수 세기
    short_word_count = sum(1 for token in doc if len(token.text) == 1) # 길이가 1인 단어 수 세기 (노이즈 가능성 높음)
    separated_hangul_count = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ]', text)) # 한글 자모 분리 정도 감지 (ㅋ, ㄱ, ㅏ 등)
    irregular_case_count = sum(1 for i in range(1, len(text)) if text[i-1].islower() and text[i].isupper()) # 불규칙한 대소문자 패턴 (소문자 다음 대문자)
    
    # 품사 태그 다양성 (낮을수록 텍스트 단조롭고 노이즈 가능성 높을 것)
    pos_counts = {}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    pos_diversity = len(pos_counts)
    
    # 노이즈 점수 계산
    noise_score = (
        (special_char_count * 2) +
        (non_korean_char_count * 1.5) +
        (short_word_count * 1.5) +
        (separated_hangul_count * 3) +
        (irregular_case_count * 2)
    ) / max(1, word_count)  # 단어 수로 나누어 정규화
    
    # 품사 다양성에 따른 보정 (다양성이 낮을수록 점수 증가)
    diversity_factor = max(1, 5 - pos_diversity) / 5
    noise_score *= (1 + diversity_factor)
    
    # 최종 점수 범위 조정 (0-100 사이)
    final_score = min(100, max(0, noise_score * 20))
    
    return final_score