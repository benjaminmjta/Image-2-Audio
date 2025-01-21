from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    app.config['IMAGE_FOLDER'] = os.path.join('.', 'app', 'static', 'images')
    app.config['AUDIO_FOLDER'] = os.path.join('.', 'app', 'static', 'audios')

    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app