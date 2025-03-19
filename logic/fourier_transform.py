import time
import numpy as np
import math

def apply_window(signal, window_type = 0):
    '''
    Apply a window to the signal to reduce spectral leakage.
    :param signal: array of signal values
    :param window_type: use hamming-window(0), np-hamming-window(1), np-hann-window(2)
    :return: windowed signal
    '''
    n = len(signal)
    match window_type:
        case 0:
            window = [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]
        case 1:
            window = np.hamming(n)
        case 2:
            window = np.hanning(n)
        case _:
            window = np.hamming(n)
    return signal * window

def zero_pad(signal, target_length, use_numpy = True):
    '''
    Zero-Pad the signal to target length.
    :param signal: array of signal values
    :param target_length: legth after zero-padding
    :param use_numpy: use numpy
    :return: zero-padded signal
    '''
    n = len(signal)
    if n >= target_length:
        raise ValueError("signal is already longer than target length")
    if use_numpy:
        return np.pad(signal, (0, target_length - n))
    else:
        signal = list(signal)
        return signal + [0] * (target_length - n)

def prepare_signal(signal, window_type = 0):
    '''
    Prepare the signal by applying a window and zero-padding.
    :param signal: array of signal values
    :param window_type: use hamming-window(0), np-hamming-window(1), np-hann-window(2)
    :return: prepared signal
    '''

    match window_type:
        case 0:
            windowed_signal = apply_window(signal, window_type)
            n = len(signal)
            target_length = 1 << (n - 1).bit_length()
            windowed_padded_signal = zero_pad(windowed_signal, target_length, use_numpy = False)
            return windowed_padded_signal
        case _:
            windowed_signal = apply_window(signal, window_type)
            n = len(signal)
            target_length = 1 << (n - 1).bit_length()
            windowed_padded_signal = zero_pad(windowed_signal, target_length, use_numpy = True)
            return windowed_padded_signal


def npft(signal):
    '''
    calculate fourier transform of signal with numpy library
    :param signal: array of signal values
    :return: array of complex values
    '''
    return np.fft.fft(signal)

def dft(signal):
    '''
    calculate discrete fourier transform of signal
    :param signal: array of signal values
    :return: array of complex values
    '''
    start_time = time.time()

    N = len(signal)
    result = []
    for k in range(N):
        r = 0
        i = 0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            r += signal[n] * np.cos(angle)
            i -= signal[n] * np.sin(angle)
        result.append(complex(r, i))

    end_time = time.time()
    print("symbol time cost:", end_time - start_time)

    return np.array(result)


def fft(signal):
    '''
    fast fourier transform on signal
    :param signal: array of signal values
    :return: array of complex values
    '''
    n = len(signal)

    if n <= 1:
        return signal

    even = fft(signal[::2])
    odd = fft(signal[1::2])

    T = [math.e**(-2j * math.pi * k/n) * odd[k] for k in range(n//2)]
    return [even[k] + T[k] for k in range(n//2)] + \
           [even[k] - T[k] for k in range(n//2)]


def get_dominant_frequency(signal, sample_rate, ft_version):
    '''
    calculates the dominant frequency of a signal
    :param signal: array of signal values
    :param sample_rate: sample rate of the signal in Hz
    :param ft_version: use numpy-fft(2) or fft(1)
    :return: dominant frequency in Hz
    '''
    match ft_version:
        case 0:
            signal = prepare_signal(signal, 1)
            fft_result = npft(signal)
        case 1:
            signal = prepare_signal(signal, 0)
            fft_result = fft(signal)
        case 2:
            signal = prepare_signal(signal, 0)
            fft_result = dft(signal)
        case _:
            signal = prepare_signal(signal, 1)
            fft_result = npft(signal)

    n = len(signal)

    freqs = [(i * sample_rate) / n for i in range(n)]
    freqs_pos = freqs[:n//2]

    amps = [abs(x) for x in fft_result]
    amps_pos = amps[:n//2]

    max_index = amps_pos.index(max(amps_pos))
    dominant_frequency = abs(freqs_pos[max_index])

    return dominant_frequency
