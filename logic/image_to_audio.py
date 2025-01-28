import math
import numpy as np
import wave
from PIL import Image
from logic import fourier_transform as ft


def image_to_audio(input, output, color_depth = 2, sample_rate = 44100, symbol_duration = 0.01, freq_rate = 100, startmarker_frequency = 2200, startmarker_duration = 0.3):
    '''
    Creates an audio file from an image file.
    :param input:Image file path and filename.
    :param output: Audio file path and filename.
    :param color_depth: Color depth of the image. Min: 2, Max: 255. Default: 2.
    :param sample_rate: Sample rate of the audio file. Default: 44100.
    :param symbol_duration: Duration of a symbol (4bits) in seconds. Default: 0.01.
    :param freq_rate: Frequency distance between the symbols in Hz. Default: 100.
    :param startmarker_frequency: Frequency of the startmarker in Hz. Default: 2200.
    :param startmarker_duration: Duration of the startmarker in seconds. Default: 0.3.
    :return: None
    '''

    bits = img2bit(input, color_depth)
    bit2audio(bits, output, sample_rate, symbol_duration, freq_rate, startmarker_duration, startmarker_frequency)

def audio_to_image(input, output, ft_version = 0, sample_rate = 44100, symbol_duration = 0.01, freq_rate = 100, startmarker_frequency = 2200, startmarker_duration = 0.3):
    '''
    Creates an recovered image from an audio file.
    :param input: Audio file path and filename
    :param output: Image file path and filename
    :param ft_version: Use numpy-fft(0), fft(1) or dft(2) for frequency analysis. Default: 0.
    :param sample_rate: Sample rate of the audio file. Default: 44100.
    :param symbol_duration: Duration of a symbol (4bits) in seconds. Default: 0.01.
    :param freq_rate: Frequency distance between the symbols in Hz. Default: 100.
    :param startmarker_frequency: Frequency of the startmarker in Hz. Default: 2200.
    :param startmarker_duration: Duration of the startmarker in seconds. Default: 0.3.
    :return: None
    '''
    bits = audio2bit(input, ft_version, symbol_duration, sample_rate, freq_rate, startmarker_frequency, startmarker_duration)
    bit2img(bits, output)

def img2bit(input_image, color_depth):
    '''
    converts image to bitstring
    :param input_image: path and filename of the input image
    :param color_depth: depth of color information min: 2 max: 255
    :return: bitstring of the image
    '''
    if color_depth not in [1, 2, 3, 4, 8]:
        raise ValueError("error creating bitstring: color depth must be 1, 2, 3, 4 or 8")

    if color_depth <= 3:
        image = Image.open(input_image).convert('L')
    else:
        image = Image.open(input_image).convert('RGB')

    width, height = image.size
    if width * height > 256 * 256:
        raise ValueError("error creating bitstring: images over 256x256 are not supported")

    height_bits = format(height, '016b')
    width_bits = format(width, '016b')
    color_depth_bits = format(color_depth, '08b')

    pixel_bits = ''
    pixels = image.getdata()

    if color_depth <= 3:
        max_pixel_value = 2 ** color_depth - 1
        pixel_bits = ''.join(
            format(round(pixel * max_pixel_value / 255), f'0{color_depth}b') for pixel in pixels
        )
    else:
        bits_per_channel = int(color_depth / 4)
        max_channel_value = int(2 ** (color_depth / 4) - 1)
        for r, g, b in pixels:
            r_bits = format(round(r * max_channel_value / 255), f'0{bits_per_channel}b')
            g_bits = format(round(g * max_channel_value / 255), f'0{bits_per_channel}b')
            b_bits = format(round(b * max_channel_value / 255), f'0{bits_per_channel}b')

            brightness = round(((r + g + b) / 3) * max_channel_value / 255)
            brightness_bits = format(brightness, f'0{bits_per_channel}b')

            pixel_bits += f'{r_bits}{g_bits}{b_bits}{brightness_bits}'

    bitstring = height_bits + width_bits + color_depth_bits + pixel_bits
    print(f"Image {input_image} successfully encoded to bitstring.")
    return bitstring


def bit2audio(bitstring, encoded_audio, sample_rate, bitgroup_duration, freq_rate, startmarker_duration, startmarker_frequency):
    '''
    converts bitstring to audio (.wav)
    :param bitstring: bitstring of the image, has to be formatted as follows: height_bits(16bits) + width_bits(16bits) + color_depth_bits(8bits) + pixel_bits
    :param encoded_audio: path and filename of the output encoded audio file
    :param sample_rate: sample rate of the audio file in Hz
    :param bitgroup_duration: duration of a bitgroup in seconds
    :param freq_rate: frequency distance between the symbols in Hz
    :param startmarker_duration: duration of the startmarker in seconds
    :param startmarker_frequency: frequency of the startmarker in Hz
    :return: None
    '''
    frequencies = [500 + i * freq_rate for i in range(16)]

    # check if bitstring is groupable with group size of 4
    padding_length = (4 - (len(bitstring) % 4)) % 4  # number of padding bits
    bitstring += '0' * padding_length  # padding

    grouped_bits = [bitstring[i:i + 4] for i in range(0, len(bitstring), 4)]
    symbols = [int(bits, 2) for bits in grouped_bits]

    amplitude = 32767  # max amp 16 bit PCM
    samples_per_group = bitgroup_duration * sample_rate
    signal = []

    # startmaker
    marker_samples = startmarker_duration * sample_rate
    marker_frequency = startmarker_frequency
    for i in range(int(marker_samples)):
        signal_value = amplitude * math.sin(2 * math.pi * marker_frequency * i / sample_rate)
        signal.append(int(signal_value))

    # symbols
    for symbol in symbols:
        frequency = frequencies[symbol]
        for i in range(int(samples_per_group)):
            signal_value = amplitude * math.sin(2 * math.pi * frequency * i / sample_rate)
            signal.append(int(signal_value))

    with wave.open(encoded_audio, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 Bytes = 16 Bit
        wf.setframerate(sample_rate)  # sample rate

        for sample in signal:
            wf.writeframes(sample.to_bytes(2, 'little', signed=True))

        print(f"image successfully encoded to {encoded_audio}")


def find_startmarker(signal, sample_rate, startmarker_frequency, startmarker_duration):
    '''
    finds startmarker in signal
    :param signal: array of signal values
    :param sample_rate: sample rate of the signal in Hz
    :param startmarker_frequency: frequency of the startmarker in Hz
    :param startmarker_duration: duration of the startmarker in seconds
    :return: index of the startmarker in the signal, -1 if no startmarker found
    '''
    window_samples = int(startmarker_duration * sample_rate)

    for i in range(0, len(signal) - window_samples, window_samples // 2):
        segment = signal[i:i + window_samples]

        fft_result = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), 1 / sample_rate)
        magnitude = np.abs(fft_result)

        dominant_freq = abs(freqs[np.argmax(magnitude)])

        if abs(dominant_freq - startmarker_frequency) <= 10:
            return i + window_samples  # startindex

    return -1


def audio2bit(encoded_audio, ft_version ,symbol_duration, sample_rate, freq_rate, startmarker_frequency, startmarker_duration):
    '''
    decodes audio (.wav) to bitstring
    :param encoded_audio: path and filename of the encoded audio file
    :param symbol_duration: duration of a symbol in seconds
    :param sample_rate: sample rate of the audio file in Hz
    :param freq_rate: frequency distance of the symbols in Hz
    :param startmarker_frequency: frequency of the startmarker in Hz
    :param startmarker_duration: duration of the startmarker in seconds
    :return: bitstring of the decoded audio
    '''
    frequencies = [500 + i * freq_rate for i in range(16)]

    with wave.open(encoded_audio, 'r') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)

    if n_channels != 1 or sampwidth != 2:
        raise ValueError("wav file must be mono and 16 bit")
    if framerate != sample_rate:
        raise ValueError("sample rate of wav file must be 44100 Hz.")

    signal = np.frombuffer(audio_data, dtype=np.int16)

    start_index = find_startmarker(signal, sample_rate, startmarker_frequency, startmarker_duration)
    print(f"startmarker found, start at {start_index} samples or {start_index / sample_rate} seconds.")
    if start_index == -1:
        raise ValueError(
            "startmarker not found, audio cannot be decoded."
        )

    samples_per_symbol = int(sample_rate * symbol_duration)

    bitstring = ""

    for i in range(start_index, len(signal), samples_per_symbol):
        segment = signal[i:i + samples_per_symbol]

        dominant_frequency = ft.get_frequency(segment, sample_rate, ft_version)

        # round to nearest frequency in frequencies if distance is < 50
        closest = min(frequencies, key=lambda x: abs(x - dominant_frequency))
        if abs(dominant_frequency - closest) < 50:
            dominant_frequency = closest

        if dominant_frequency in frequencies:
            bitstring += format(int(dominant_frequency/freq_rate) - int(frequencies[0]/freq_rate), '04b')
        else:
            bitstring += format(int(closest/freq_rate) - int(frequencies[0]/freq_rate), '04b')
            print(f"unknown frequency: {dominant_frequency}Hz, using {closest}Hz.")

    return bitstring


def bit2img(bitstring, output_image):
    '''
    converts bitstring to image
    :param bitstring: has to be formatted as follows: height_bits(16bits) + width_bits(16bits) + color_depth_bits(8bits) + pixel_bits
    :param output_image: path and filename of the output image
    :return: None
    '''
    ## check min length
    if len(bitstring) < 40:
        raise ValueError("invalid bitstring length")

    ## get height and width
    height = int(bitstring[:16], 2)
    width = int(bitstring[16:32], 2)
    color_depth = int(bitstring[32:40], 2)

    if color_depth not in [1, 2, 3, 4, 8]:
        raise ValueError("Error decoding bitstring: color depth must be 1, 2, 3, 4 or 8")

    bits_per_pixel = color_depth
    pixel_bits = bitstring[40:]
    pixels = []

    if color_depth <= 3:
        max_pixel_value = 2 ** color_depth - 1
        pixels = [
            int(pixel_bits[i : i + bits_per_pixel], 2) * (255 // max_pixel_value)
            for i in range(0, len(pixel_bits), bits_per_pixel)
        ]
        mode = 'L'
    else:
        bits_per_channel = int(color_depth / 4)
        max_channel_value = int(2 ** (color_depth / 4) - 1)

        for i in range(0, len(pixel_bits), color_depth):
            r = int(pixel_bits[i : i + bits_per_channel], 2) * (255 // max_channel_value)
            g = int(pixel_bits[i + bits_per_channel : i + 2 * bits_per_channel], 2) * (255 // max_channel_value)
            b = int(pixel_bits[i + 2 * bits_per_channel : i + 3 * bits_per_channel], 2) * (255 // max_channel_value)
            brightness = int(pixel_bits[i + 3 * bits_per_channel : i + 4 * bits_per_channel], 2) * (255 // max_channel_value)

            r = r if brightness else r // 2
            g = g if brightness else g // 2
            b = b if brightness else b // 2

            pixels.append((r, g, b))

        mode = 'RGB'

    image = Image.new(mode, (width, height))
    image.putdata(pixels)
    image.save(output_image)

    print(f"Image successfully decoded from bitstring to {output_image}")