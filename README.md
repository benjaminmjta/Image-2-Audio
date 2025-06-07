# Image Over Audio

This Python application allows you to encode images into audio signals and recover images from audio files. The image pixel data is translated into a bitstring and converted to a .wav audio file, where each group of bits is represented by a unique frequency.

## Features

- Upload an image and convert it to a .wav audio file
- Upload or record an audio file and recover the encoded image
- Adjustable color depth and image size
- Supports both grayscale and color images
- Web interface built with Flask

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/benjaminmjta/ImageToAudio.git
   cd ImageToAudio
   ```

2. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the application:

   ```
   python app.py
   ```

4. Open your browser and go to `http://localhost:8000`

## Usage

- Use the web interface to upload an image and convert it to audio.
- Upload or record an audio file to recover the image.
- Adjust color depth and other parameters as needed.

