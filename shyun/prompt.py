RANDOM = 42

def extract_label_prompt():
    sys_prompt = '''당신은 기사 제목을 보고 어떤 분야의 기사인지 맞추는 전문가입니다.
[지시사항]
1. 주어진 데이터는 개행 기호(\\n)로 구분된 뉴스 기사 제목들입니다.
2. 주어진 데이터에는 임의의 자리에 문맥에 맞지 않는 글자가 무작위로 삽입되어 있습니다.
3. 해당 데이터들을 가장 잘 표현하는 포괄적인 기사 분야를 한 단어로만 출력하세요.'''

    fewshot = [
        {"role": "user", "content": "신간2문학A음악이 q야기ji"},
        {"role": "assistant", "content": "문화"},
        {"role": "user", "content": "정부 4월 한반도 위설d근거O다Y되J Q6야d합"},
        {"role": "assistant", "content": "정치"},
        {"role": "user", "content": "충북교육청 노h조합mYfR동K군별 4양한 목소OH용"},
        {"role": "assistant", "content": "사회"},
        {"role": "user", "content": "국제k호단체HrUTs 유엔 테단체 1정 움직f에k우려"},
        {"role": "assistant", "content": "해외"}
    ]

    return sys_prompt, fewshot

def clean_text_prompt(key):
    sys_prompt = f'''당신은 기사 제목을 복원하는 전문가입니다.
1. 키워드는 '{key}'입니다.
2. 어색한 문맥을 키워드를 참고하여 올바르게 고치세요.
3. 복원이 안 되거나 일부만 가능할 경우, 해당 키워드를 포함한 형태로 대체하세요.'''

    fewshot = [
        {"role": "user", "content": "해외로밍 m금폭탄 n동차단 더 빨진다"},
        {"role": "assistant", "content": "해외로밍 요금 폭탄 자동차단 더 빨라진다"},
        {"role": "user", "content": "코z나0i9규직Zj정규직 E문R비교"},
        {"role": "assistant", "content": f"{key} 코로나 정규직 비정규직 E문R비교"},
        {"role": "user", "content": "정i 파1 미사z KT 이용기간 2e 단 Q분종U2보"},
        {"role": "assistant", "content": f"{key} 정i 파1 미사z KT 이용기간 2e 단 Q분종U2보"}
    ]

    return sys_prompt, fewshot

def clean_label_prompt(keys):
    keys_str = ', '.join(keys)

    sys_prompt = f'''기사 제목을 보고 주어진 분야 중 올바른 분야의 번호를 선택하세요.
    - 기사 분야: {keys_str}
    - 위 분야 중 올바른 분야의 번호를 선택하세요.
    - 문맥적으로 올바르지 않은 데이터는 '불가'라고 출력하세요.
    - 반드시 주어진 키워드 중 한 개만 선택하세요.'''

    fewshot = [
        {"role": "user", "content": "구미 영천 등 경북 9개 시군 폭염주의보"},
        {"role": "assistant", "content" : "0"},
        {"role": "user", "content": "울산 진보3당 북구시설관리공단 설립 조례안 철회하라"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "국제유가곡물가올해 농업경영비 33 감소 전망"},
        {"role": "assistant", "content": "3"}
    ]

    return sys_prompt, fewshot

def generate_prompt(key, shots):
    sys_prompt = f'''당신은 '{key}' 분야의 기사 제목을 작성하는 전문 작가입니다. 주어진 예시를 바탕으로 다양한 형태의 기사 제목을 작성해 주세요.'''

    fewshot = []
    for shot in shots:
        fewshot.append({"role": "user", "content": f"'{key}' 분야에 해당하는 기사 제목을 한 개만 생성하세요."})
        fewshot.append({"role": "assistant", "content": shot})
    
    return sys_prompt, fewshot

def regenerate_prompt(key):
    sys_prompt = f'''당신은 '{key}' 분야의 기사 제목을 재작성하는 전문가입니다. 제공된 기사 제목의 의미를 보존하며 재작성 하세요.'''

    fewshot = [
        {'role': 'user', 'content': '보령소식 보령시 시간선택제 공무원 3명 모집'},
        {'role': 'assistant', 'content': '보령시에서 새로운 시간제 공무원 모집'}
    ]

    return sys_prompt, fewshot