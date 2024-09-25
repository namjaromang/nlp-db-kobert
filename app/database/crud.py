# app/database/crud.py
import json
from sqlalchemy.orm import Session
from app.database.models import AnalysisResult, Article, Category
from app import db


# 데이터 저장
def save_analysis_result(text, label):
    label_json = json.dumps(label, ensure_ascii=False)  # label을 JSON 문자열로 변환
    result = AnalysisResult(text=text, label=label_json)
    db.session.add(result)
    db.session.commit()


# 데이터 조회
def get_all_results():
    return AnalysisResult.query.all()


def get_valid_categories():
    # naver_code가 99가 아닌 카테고리들을 가져오기
    return Category.query.filter(Category.naver_code != 99).all()

def get_articles_by_non_miscellaneous():
    # category_id가 99가 아닌 기사들을 가져오기
    return Article.query.filter(Article.category_id != 99).all()

def get_articles_by_category(category_id):
    # 특정 카테고리의 기사들을 가져오기
    return Article.query.filter(Article.category_id == category_id).all()


def update_article_category(article_id, new_category):
    # 새로운 카테고리로 기사 업데이트
    article = Article.query.get(article_id)
    if article:
        # 새로운 카테고리 ID로 업데이트 (new_category는 카테고리명에서 id로 변환해야 함)
        category_id = db.session.query(Category.id).filter_by(title=new_category).first()
        if category_id:
            article.category_id = category_id[0]
            db.session.commit()
