from flask import Blueprint, render_template, request, send_file, url_for
import os
from logic import image_to_audio as ita

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    filename = file.filename
    file_path = os.path.join('.', 'app', 'static', 'images', filename)
    file.save(file_path)

    return render_template(
        'index.html',
        uploaded_image = url_for('static', filename= os.path.join('images', filename)),
        image_name = filename
    )

@main.route('/convert_to_audio', methods=['POST'])
def convert_image_to_audio():
    color_depth = int(request.form['color_depth'])
    image_name = request.form['image_name']
    image_path = os.path.join('.', 'app', 'static', 'images', image_name)

    if not os.path.exists(image_path):
        return 'image not found', 404

    audio_name = os.path.splitext(image_name)[0] + '.wav'
    audio_path = os.path.join('.', 'app', 'static', 'audios', audio_name)

    ita.image_to_audio(image_path, audio_path, color_depth)

    return render_template(
        'index.html',
        uploaded_image = url_for('static', filename= os.path.join('images', image_name)),
        image_name = image_name,
        audio_file = url_for('static', filename= os.path.join('audios', audio_name))
    )

@main.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return 'no file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'no file selected', 400
    filename = file.filename
    file_path = os.path.join('.', 'app', 'static', 'audios', filename)
    file.save(file_path)

    return render_template(
        'index.html',
        uploaded_audio = url_for('static', filename= os.path.join('audios', filename)),
        audio_name = filename
    )

@main.route('/recover_image', methods=['POST'])
def recover_image():
    audio_name = request.form['audio_name']
    audio_path = os.path.join('.', 'app', 'static', 'audios', audio_name)

    if not os.path.exists(audio_path):
        return 'audio not found', 404

    image_name = os.path.splitext(audio_name)[0] + '_recovered.png'
    image_path = os.path.join('.', 'app', 'static', 'images', image_name)

    ita.audio_to_image(audio_path, image_path)

    return render_template(
        'index.html',
        uploaded_audio = url_for('static', filename= os.path.join('audios', audio_name)),
        audio_name = audio_name,
        recovered_image = url_for('static', filename= os.path.join('images', image_name)),
        image_name = image_name
    )