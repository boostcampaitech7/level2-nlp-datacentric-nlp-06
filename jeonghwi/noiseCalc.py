def calculate_ascii_noise(text):
    # 문장에 대한 ascii noise 계산
    total_chars = len(text)
    ascii_chars = 0 
    for ch in text:
        if ch == " ":
            continue
        if ord(ch) < 128:
            ascii_chars+=1
            
    non_ascii_chars = total_chars - ascii_chars  # 비-ASCII 문자 개수
    
    ascii_ratio = (ascii_chars / total_chars) * 100  # ASCII 문자 비율 (%)
    
    return ascii_chars, non_ascii_chars, ascii_ratio

def calculate_ascii_noise_bulk(texts):
    # 여러개의 단어에 대한 ascii noise 계산
    asc_counts = []; non_asc_counts = []; percentages = []

    for text_with_noise in texts:
        ascii_count, non_ascii_count, ascii_percentage = calculate_ascii_noise(text_with_noise)
        ascii_percentage = round(ascii_percentage,2)
        asc_counts.append(ascii_count)
        non_asc_counts.append(non_ascii_count)
        percentages.append(ascii_percentage)

    return asc_counts, non_asc_counts, percentages

def get_noise_df(df,percent,types):
    # types == ["up","down","middle"]
    # 특정 노이즈 % 이상, 이하, 중간에 대한 data return
    text_datas = df["text"]
    asc_counts, non_asc_counts, ratios = calculate_ascii_noise_bulk(text_datas)
    df["ratio"] = ratios
    if types == "up" # n% 초과
        return df[df["ratio"]>percent]

    elif types == "down" # n% 미만
        return df[df["ratio"]<percent]
    
    elif types == "middle": # n[0]% 초과 n[1]% 미만
        down_df = df[df["ratio"]<percent[1]]
        result_df = down_df[down_df["ratio"]>percent[0]]
        return result_df
