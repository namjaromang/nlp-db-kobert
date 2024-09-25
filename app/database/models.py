import datetime

import pytz
from sqlalchemy import Column, Integer, String, Text
from app import db


class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    label = Column(Text, nullable=False)  # label을 JSON 문자열로 저장

    def __init__(self, text, label):
        self.text = text
        self.label = label


class Article(db.Model):
    __tablename__ = 'articles'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    title = db.Column(String(255), nullable=False)
    publisher = db.Column(String(255), nullable=False)
    original_link = db.Column(String(500), nullable=True)
    naver_link = db.Column(String(500), nullable=True)
    description = db.Column(String(2000), nullable=True)
    category_id = db.Column(db.BigInteger, nullable=False)
    pub_date = db.Column(db.DateTime, nullable=False)
    create_date = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(pytz.timezone('Asia/Seoul')))


class Category(db.Model):
    __tablename__ = 'category'

    id = db.Column(db.BigInteger, primary_key=True)  # BigInteger로 수정
    title = db.Column(db.String(100), nullable=False)
    naver_code = db.Column(db.Integer, nullable=False)  # 크기 제한 제거
    limit = db.Column(db.Integer, nullable=False)  # 크기 제한 제거
    create_date = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(pytz.timezone('Asia/Seoul')))
