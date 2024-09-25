from kiwipiepy import Kiwi
from transformers import BertTokenizer, BertForSequenceClassification

# KoBERT 모델 및 토크나이저 로드
model_name = "monologg/kobert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=8)  # 카테고리 확장

# Kiwi 로드 (한국어 형태소 분석)
kiwi = Kiwi()


def extract_keywords(text):
    # Kiwi를 사용한 핵심 단어 추출
    tokens = kiwi.analyze(text)
    # 추출된 형태소 중에서 명사(N)만 추출
    keywords = [word[0] for word in tokens[0][0] if 'NNG' in word[1] or 'NNP' in word[1]]
    return keywords