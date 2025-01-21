import numpy as np
import math

def dft(signal):
    '''
    calculate discrete fourier transform of signal
    :param signal: array of signal values
    :return: array of complex values
    '''
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
    return np.array(result)


def fft(signal):
    '''
    fast fourier transform on signal
    :param signal: dataarray of signal values
    :return: dataarray of complex values
    '''

    n = len(signal)

    if n <= 1:
        return signal

    even = fft(signal[::2])
    odd = fft(signal[1::2])

    T = [math.e**(-2j * math.pi * k/n) * odd[k] for k in range(n//2)]
    return [even[k] + T[k] for k in range(n//2)] + \
           [even[k] - T[k] for k in range(n // 2)]


def get_frequency(signal, sample_rate, ft_version):
    '''
    calculates the dominant frequency of a signal
    :param signal: dataarray of signal values
    :param sample_rate: sample rate of the signal in Hz
    :param ft_version: use fft(1) or dft(0) type:bool
    :return: dominant frequency in Hz
    '''

    n = len(signal)

    if n%2 != 0:
        new_length = 2 ** int(math.floor(math.log2(n)))
        signal = signal[:new_length]

    n = len(signal)

    if ft_version == 0:
        fft_result = dft(signal)
    else:
        fft_result = fft(signal)

    amps = [abs(x) for x in fft_result]
    freqs = [(i * sample_rate) / n for i in range(n)]

    freqs_pos = freqs[:n//2]
    amps_pos = amps[:n//2]

    max_index = amps_pos.index(max(amps_pos))
    dominant_frequency = abs(freqs_pos[max_index])

    return dominant_frequency