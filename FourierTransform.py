import numpy as np
import plotly.graph_objects as go

# Parameter für das Signal
sampling_rate = 100  # Abtastrate in Hz
duration = 1  # Dauer des Signals in Sekunden
frequency = 15  # Frequenz des Signals in Hz

# Zeitachse erstellen
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Signal generieren (z. B. Sinuswelle)
signal = np.sin(2 * np.pi * frequency * t)

N = sampling_rate * duration
result = []
for k in range(N):
    r = 0
    i = 0
    for n in range(N):
        angle = 2 * np.pi * k * n / N
        print('angle1 '+ angle)
        angle2 = 2 * np.pi * n / N
        print(angle2)
        '''
        k = frequenz
        n = bogenmaß
        N = Abtastrate * Dauer = Umfang
        radius = U / (2pi)
        winkel = bogenmaß / radius
        winkel = 2pi * n / N
        wird verstärkt mit k
        '''
        r += signal[n] * np.cos(angle)
        i -= signal[n] * np.sin(angle)
    result.append(complex(r, i))

for i in range(len(result)):
    print(f'{i} : {result[i]}')