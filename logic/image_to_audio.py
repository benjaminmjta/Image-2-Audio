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

def audio_to_image(input, output, sample_rate = 44100, symbol_duration = 0.01, freq_rate = 100, startmarker_frequency = 2200, startmarker_duration = 0.3):
    '''
    Creates an recovered image from an audio file.
    :param input: Audio file path and filename
    :param output: Image file path and filename
    :param sample_rate: Sample rate of the audio file. Default: 44100.
    :param symbol_duration: Duration of a symbol (4bits) in seconds. Default: 0.01.
    :param freq_rate: Frequency distance between the symbols in Hz. Default: 100.
    :param startmarker_frequency: Frequency of the startmarker in Hz. Default: 2200.
    :param startmarker_duration: Duration of the startmarker in seconds. Default: 0.3.
    :return: None
    '''
    bits = audio2bit(input, symbol_duration, sample_rate, freq_rate, startmarker_frequency, startmarker_duration)
    bit2img(bits, output)

def img2bit(input_image, color_depth):
    '''
    converts image to bitstring
    :param input_image: path and filename of the input image
    :param color_depth: depth of color information min: 2 max: 255
    :return: bitstring of the image
    '''
    ## img to binary (black white) umwandeln
    image_convert_format = 'L'
    if color_depth <= 2: image_convert_format = '1'
    image = Image.open(input_image).convert(f'{image_convert_format}')  # 'L' = grayscale '1' = black and white

    if not (2 <= color_depth <= 255):
        raise ValueError("Color depth must be between 2 and 255")

    color_depth_bits = format(color_depth, '08b')

    bits_per_pixel = math.ceil(math.log2(color_depth))

    ## width and length
    width, height = image.size
    if width * height > 256 * 256:
        raise ValueError("images over 256x256 are not supported")
    height_bits = format(height, '016b')
    width_bits = format(width, '016b')

    ## img to binary bitstring
    ## 0 -> white, 255 -> black
    pixels = image.getdata()
    max_pixel_value = 2 ** bits_per_pixel - 1
    pixels_scaled = [round(pixel * max_pixel_value / 255) for pixel in pixels]

    # Convert pixel values to binary representation
    pixels_bits = ''.join(format(pixel, f'0{bits_per_pixel}b') for pixel in pixels_scaled)

    ## create final string
    bitstring = height_bits + width_bits + color_depth_bits + pixels_bits

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

    # iterate through signal to find startmarker
    for i in range(0, len(signal) - window_samples, window_samples // 2):
        segment = signal[i:i + window_samples]

        # fourier transform
        fft_result = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), 1 / sample_rate)
        magnitude = np.abs(fft_result)

        # get dominant freq
        dominant_freq = abs(freqs[np.argmax(magnitude)])

        # check if dominant freq = startmarker
        if abs(dominant_freq - startmarker_frequency) <= 10:
            return i + window_samples  # startindex

    return -1  # no startmarker found


def audio2bit(encoded_audio, symbol_duration, sample_rate, freq_rate, startmarker_frequency, startmarker_duration):
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
    frequencies.append(startmarker_frequency)

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

    freqs = np.fft.fftfreq(samples_per_symbol, 1 / sample_rate)

    # decode every symbol
    for i in range(start_index, len(signal), samples_per_symbol):
        # get current symbol
        segment = signal[i:i + samples_per_symbol]

        '''
        # fourier transform manual implementation
        # 1 -> fft, 0 -> dft
        ft_version = 1
        dominant_frequency = ft.get_frequency(segment, sample_rate, ft_version)
        print(f"dominant frequency manual: {dominant_frequency}Hz")
        dominant_frequency = min(frequencies, key=lambda x: abs(x - dominant_frequency))
        print(f"rounded dominant frequency manual: {dominant_frequency}Hz")
        '''

        '''
        if dominant_frequency == frequencies[16]:

            continue

        if dominant_frequency in frequencies:
            bitstring += format(int(dominant_frequency/freq_rate) - int(frequencies[0]/freq_rate), '04b')
        else:
            print(f"unknown frequency: {dominant_frequency}Hz, skipped.")
        '''

        # fast fourier via np library
        # fourier transform to get current frequency
        fft_result = np.fft.fft(segment)

        # get dominating freq
        magnitude = np.abs(fft_result)
        dominant_freq = abs(freqs[np.argmax(magnitude)])

        # freq to symbol
        if dominant_freq in frequencies:
            bitstring += format(int(dominant_freq / freq_rate) - int(frequencies[0] / freq_rate), '04b')
        else:
            print(f"unknown frequency: {dominant_freq}Hz, skipped.")

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

    bits_per_pixel = math.ceil(math.log2(color_depth))

    ## check if bitstring length fits with proportions
    if len(bitstring) < height * width * bits_per_pixel + 16 + 16 + 8:
        raise ValueError("invalid bitstring length")

    # Extract pixel bits and convert them to grayscale values
    pixel_bits = bitstring[40:]
    pixels = [
        int(pixel_bits[i:i + bits_per_pixel], 2) * (255 // (2 ** bits_per_pixel - 1))
        for i in range(0, len(pixel_bits), bits_per_pixel)
    ]

    image_convert_format = 'L'
    if color_depth <= 2: image_convert_format = '1'

    # Create a new image
    image = Image.new(f'{image_convert_format}', (width, height))
    image.putdata(pixels)

    # Save the new image
    image.save(output_image)