import math
import base64 as b64
import numpy as np
import wave
from PIL import Image
from scipy.io.wavfile import write

# converts image (black white) to binary string
def img2bit(input_image):
    ## img to binary (black white) umwandeln
    image = Image.open(input_image).convert('1')  # '1' = black white

    ## width and length
    width, height = image.size
    if width * height > 256 * 256:
        raise ValueError("images over 256x256 are not supported")
    height_bits = format(height, '016b')
    width_bits = format(width, '016b')

    ## img to binary bitstring
    ## 0 -> white, 255 -> black
    pixels = image.getdata()
    pixels_bits = ''.join(['1' if pixel > 127 else '0' for pixel in pixels])

    ## create final string
    bitstring = height_bits + width_bits + pixels_bits
    return bitstring



def bit2audio(bitstring, encoded_audio, sample_rate = 44100, bitgroup_duration = 0.01):
    frequencies = [1000 + i * 1000 for i in range(16)]

    # check if bitstring is groupable with group size of 4
    padding_length = (4 - (len(bitstring) % 4)) % 4  # number of padding bits
    bitstring += '0' * padding_length  # padding

    grouped_bits = [bitstring[i:i + 4] for i in range(0, len(bitstring), 4)]
    symbols = [int(bits, 2) for bits in grouped_bits]

    amplitude = 32767 # max amp 16 bit PCM
    samples_per_group = bitgroup_duration * sample_rate
    signal = []

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

        print(f"image successfully converted via FSK into: {encoded_audio}")


def audio2bit(encoded_audio, bitgroup_duration = 0.01, sample_rate = 44100):
    frequencies = [1000 + i * 1000 for i in range(16)]

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

    samples_per_symbol = int(sample_rate * bitgroup_duration)

    bitstring = ""

    # decode every symbol
    for i in range(0, len(signal), samples_per_symbol):
        # get current symbol
        segment = signal[i:i+samples_per_symbol]

        # fourier transform to get current frequency
        fft_result = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), 1 / sample_rate)

        # get dominating freq
        magnitude = np.abs(fft_result)
        dominant_freq = freqs[np.argmax(magnitude)]

        # freq to symbol
        if abs(dominant_freq) in frequencies:
            bitstring += format(int(abs(dominant_freq)/1000)-1, '04b')
        else:
            print(f"unknown frequency: {dominant_freq}Hz, skipped.")

    return bitstring


# converts bitstring to image with pillow
def bit2img(bitstring, output_image):
    ## check min length
    if len(bitstring) < 32:
        raise ValueError("invalid bitstring length")

    ## get height and width
    height = int(bitstring[:16], 2)
    width = int(bitstring[16:32], 2)

    ## check if bitstring length fits with proportions
    if len(bitstring) < height * width + 32:
        raise ValueError("invalid bitstring length")

    # bit to pixels (0 = Schwarz, 255 = Weiß)
    pixels = [255 if bit == '1' else 0 for bit in bitstring[32:(width*height+32)]] # padding not included

    # create new image
    image = Image.new('1', (width, height))
    image.putdata(pixels)

    # save new image
    image.save(output_image)
    print(f"image successfully encoded to {output_image}")



input_image = 'skibidi_gembris_lowres.png'
output_image = 'skibidy_gembris_lowres_received.png'
encoded_audio = 'image_encoded.wav'
bits = img2bit(input_image)
print(bits)
# bit2audio(bits, encoded_audio)
new_bits = audio2bit(encoded_audio)
print(new_bits)
bit2img(new_bits, output_image)