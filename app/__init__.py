# app/__init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.config import Config

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    # 데이터베이스 초기화
    with app.app_context():
        db.create_all()

    from app.routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    return app
