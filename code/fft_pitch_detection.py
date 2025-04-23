import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load audio sample
rate, data = wavfile.read("../audio/piano.wav")
if data.ndim > 1:
    data = data[:, 0]

# Take a small window for pitch detection
window = data[0:4096]
window = window * np.hanning(len(window))

# FFT
fft_data = np.fft.fft(window)
frequencies = np.fft.fftfreq(len(window), 1/rate)

# Find peak frequency
magnitude = np.abs(fft_data)
peak_freq = np.abs(frequencies[np.argmax(magnitude)])

print(f"Estimated pitch: {peak_freq:.2f} Hz")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(window)//2], magnitude[:len(window)//2])
plt.title("Pitch Detection using FFT")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid()
plt.show()
