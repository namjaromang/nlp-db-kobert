import os
import time
from multiprocessing import Process

import matplotlib.pyplot as plt
import torch
from flask import Blueprint, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from app import create_app  # create_app 함수를 임포트
from app.database.crud import get_articles_by_category, get_valid_categories, get_articles_by_non_miscellaneous, \
    update_article_category
from app.database.models import Article
from app.models.text_processing import extract_keywords

bp = Blueprint('routes', __name__)  # Flask의 Blueprint를 사용
# 모델 및 토크나이저 로드
model_name = "./trained_model"  # 학습한 모델 경로
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()  # 모델을 평가 모드로 설정
app = create_app()


@bp.route('/train', methods=['GET'])
def train_model():
    categories = get_valid_categories()

    if not categories:
        return jsonify({"error": "No valid categories found"}), 404

    training_data = []
    labels = []
    category_map = {category.id: idx for idx, category in enumerate(categories)}  # 카테고리를 인덱스로 매핑

    for category in categories:
        articles = get_articles_by_category(category.id)

        for article in articles:
            text = article.title
            keywords = extract_keywords(text)  # 제목에서 주요 키워드 추출

            # 키워드를 기반으로 학습 데이터 생성
            if keywords:
                training_data.append(' '.join(keywords))  # 추출된 키워드를 사용하여 텍스트 구성
                labels.append(category_map[category.id])  # 카테고리 ID를 인덱스로 변환하여 라벨로 사용

    if not training_data:
        return jsonify({"error": "No training data found after keyword extraction"}), 404

    # 학습 데이터를 토크나이징 및 준비
    model_name = "monologg/kobert"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    encodings = tokenizer(training_data, truncation=True, padding=True, max_length=512, return_tensors="pt")
    labels = torch.tensor(labels)

    # 데이터셋 생성
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    dataset = NewsDataset(encodings, labels)

    # KoBERT 모델 설정
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(category_map))

    # 학습 설정
    training_args = TrainingArguments(
        output_dir='./results',  # 모델이 저장될 디렉토리 설정
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # 모델 학습
    trainer.train()

    # 모델 저장
    trainer.save_model("./trained_model")  # 학습된 모델을 './trained_model' 경로에 저장
    tokenizer.save_pretrained("./trained_model")  # 토크나이저 저장

    return jsonify({"message": "Training complete, model saved."})


def reclassify_worker(articles, category_map):
    # 각 프로세스에서 Flask 애플리케이션을 생성
    app = create_app()

    # 애플리케이션 컨텍스트 설정
    with app.app_context():
        for article in articles:
            text = article.title

            # 키워드 추출
            keywords = extract_keywords(text)
            if not keywords:
                continue  # 키워드가 없으면 건너뜀

            # 추출된 키워드를 모델에 입력하기 위해 공백으로 연결
            input_text = ' '.join(keywords)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

            # 모델 예측
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_label = torch.argmax(logits, dim=1).item()

            # 예측된 라벨에 따라 카테고리 업데이트
            if predicted_label in category_map:
                new_category = category_map[predicted_label]
                update_article_category(article.id, new_category)


@bp.route('/reclassify', methods=['GET'])
def reclassify_articles():
    articles = get_articles_by_non_miscellaneous()

    if not articles:
        return jsonify({"error": "No articles found for reclassification"}), 404

    # DB에서 카테고리 맵을 가져오기
    categories = get_valid_categories()
    category_map = {category.id: category.title for category in categories}
    category_map[0] = "예외"

    # 데이터를 각 프로세스에 분할하여 전달
    num_processes = os.cpu_count()  # CPU 코어 수에 맞춰 프로세스 생성
    chunk_size = len(articles) // num_processes
    processes = []

    # 각 프로세스에 분할된 데이터를 전달, 각 프로세스에서 Flask 앱 생성
    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = None if i == num_processes - 1 else (i + 1) * chunk_size
        article_chunk = articles[start_index:end_index]
        p = Process(target=reclassify_worker, args=(article_chunk, category_map))
        processes.append(p)
        p.start()

    # 모든 프로세스가 완료될 때까지 대기
    for p in processes:
        p.join()

    return jsonify({"message": "Reclassification complete."})


def test_model_on_sample_data(test_articles):
    category_map = {0: "Category A", 1: "Category B", 2: "Category C"}  # 예시 카테고리 맵

    total_articles = len(test_articles)
    correct_predictions = 0
    times = []
    correct_counts = []

    start_time = time.time()

    for idx, article in enumerate(test_articles):
        text = article.title

        # 키워드 추출
        keywords = extract_keywords(text)
        if not keywords:
            continue

        # 추출된 키워드를 모델에 입력하기 위해 공백으로 연결
        input_text = ' '.join(keywords)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # 모델 예측
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()

        # 실시간 처리율을 위해 시간을 측정
        times.append(time.time() - start_time)

        # 임의로 정답 라벨이 존재한다고 가정하고 비교
        if predicted_label == article.category_id:  # 실제 라벨과 비교하는 부분 (article.category_id)
            correct_predictions += 1

        correct_counts.append(correct_predictions / (idx + 1))

        # 실시간 그래프 업데이트
        if (idx + 1) % 10 == 0:  # 10개마다 그래프 갱신
            plot_real_time_accuracy(times, correct_counts)

    return correct_predictions / total_articles


def plot_real_time_accuracy(times, correct_counts):
    plt.clf()
    plt.plot(times, correct_counts, label="Accuracy Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Accuracy")
    plt.title("Real-Time Model Accuracy")
    plt.legend()
    plt.pause(0.01)  # 실시간 그래프 갱신 속도


if __name__ == "__main__":
    # 예시 테스트 기사 데이터
    # 1,2,3,4,5,6
    # '정치', '경제', '사회', '문화', '세계', 'IT/과학'

    test_articles = [
        Article(title="Sample title about sports", category_id=1),
        Article(title="Sample title about politics", category_id=2),
        Article(title="'세기의 이혼' 중 드러난 '노태우 300억'‥비자금으로 불린 재산?", category_id=3),
        Article(title="Sample title about politics", category_id=2),
        Article(title="트럼프, '명예훼손' 1천억 원대 배상 바이든의 마녀사냥", category_id=5),
        Article(title="조관우 2번 이혼+사기…자식들에 상처 줘 미안해", category_id=7),

        # ... 추가 데이터
    ]

    # 학습된 모델로 테스트 실행
    accuracy = test_model_on_sample_data(test_articles)
    print(f"Final accuracy on test data: {accuracy * 100:.2f}%")
