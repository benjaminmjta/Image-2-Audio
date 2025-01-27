from flask import Flask
import os
from flask_font_awesome import FontAwesome

font_awesome = FontAwesome()


def create_app():
    app = Flask(__name__)
    app.config['IMAGE_FOLDER'] = os.path.join('.', 'app', 'static', 'images')
    app.config['AUDIO_FOLDER'] = os.path.join('.', 'app', 'static', 'audios')

    font_awesome.init_app(app)

    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app