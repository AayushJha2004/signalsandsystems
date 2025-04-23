import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load audio file
rate, data = wavfile.read("../audio/piano.wav")
if data.ndim > 1:
    data = data[:, 0]  # Use only one channel if stereo

# FFT
N = len(data)
T = 1.0 / rate
yf = np.fft.fft(data)
xf = np.fft.fftfreq(N, T)[:N//2]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.grid()
plt.title("Audio Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.show()
