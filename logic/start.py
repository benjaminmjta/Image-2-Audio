import image_to_audio as ita
import fourier_transform as ft
import os

input_image = os.path.join('..', 'Files', 'Images', '64x64.png')
output_image = os.path.join('..', 'Files', 'Images', 'image_decoded.png')
audio_file = os.path.join('..', 'Files', 'Audios', 'encoded_image.wav')

color_depth = 2 # depth of color information min: 2 max: 255

sample_rate = 44100
symbol_duration = 0.01
freq_rate = 100
startmarker_frequency = 2200
startmarker_duration = 0.3

bits = ita.img2bit(input_image, color_depth)
ita.bit2audio(bits, audio_file, sample_rate, symbol_duration, freq_rate, startmarker_duration, startmarker_frequency)
new_bits = ita.audio2bit(audio_file, symbol_duration, sample_rate, freq_rate, startmarker_frequency, startmarker_duration)
ita.bit2img(new_bits, output_image)

