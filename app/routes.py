from datetime import datetime
from flask import Blueprint, render_template, request, send_file, url_for, jsonify
import os
from logic import image_to_audio as ita
import subprocess

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'no file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'no file selected', 400
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

    if color_depth > 4:
        color_depth = 8

    if color_depth not in [1, 2, 3, 4, 8]:
        return 'invalid color depth', 400

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

@main.route('/save_recording', methods=['POST'])
def save_audio():
    if 'audio' not in request.files:
        return jsonify(success = False, error = 'no audio file uploaded'), 400
    audio = request.files['audio']
    webm_path = os.path.join('.', 'app', 'static', 'audios', 'temp.webm')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'recording_{timestamp}.wav'
    filepath = os.path.join('.', 'app', 'static', 'audios', filename)

    try:
        audio.save(webm_path)
        command = ['ffmpeg', '-i', webm_path, '-ar', '44100', '-ac', '1', '-sample_fmt', 's16', filepath]
        subprocess.run(command, check=True)
        os.remove(webm_path)
        return jsonify(success = True, filename = filename)
    except Exception as e:
        print(f'error saving audio: {e}')
        return jsonify(success = False, error = str(e)), 500

@main.route('/recorded_audio')
def recorded_audio():
    filename = request.args.get('filename')
    return render_template(
        'index.html',
        uploaded_audio = url_for('static', filename= os.path.join('audios', filename)),
        audio_name = filename
    )

@main.route('/recover_image', methods=['POST'])
def recover_image():
    audio_name = request.form['audio_name']
    ft_version = int(request.form.get('ft_version', 0))
    audio_path = os.path.join('.', 'app', 'static', 'audios', audio_name)

    if not os.path.exists(audio_path):
        return 'audio not found', 404

    image_name = os.path.splitext(audio_name)[0] + '_recovered.png'
    image_path = os.path.join('.', 'app', 'static', 'images', image_name)

    try:
        ita.audio_to_image(audio_path, image_path, ft_version)

        return render_template(
            'index.html',
            uploaded_audio = url_for('static', filename= os.path.join('audios', audio_name)),
            audio_name = audio_name,
            recovered_image = url_for('static', filename= os.path.join('images', image_name)),
            image_name = image_name
        )
    except Exception as e:
        error_message = f'error recovering image from audio {audio_name}. try a different one.'
        print(f'{error_message}: {e}')
        return render_template(
            'index.html',
            uploaded_audio = url_for('static', filename= os.path.join('audios', audio_name)),
            audio_name = audio_name,
            error_message = error_message
        )