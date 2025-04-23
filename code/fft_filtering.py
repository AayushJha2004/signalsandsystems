import numpy as np
import matplotlib.pyplot as plt

# Generate signal
fs = 500
t = np.arange(0, 2.0, 1/fs)
signal = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)

# Add noise
signal += 0.5 * np.random.randn(t.size)

# FFT
fft_signal = np.fft.fft(signal)
freq = np.fft.fftfreq(len(t), d=1/fs)

# Band-pass filter between 40 and 130 Hz
filtered_fft = fft_signal.copy()
filtered_fft[(np.abs(freq) < 40) | (np.abs(freq) > 130)] = 0

# Inverse FFT
filtered_signal = np.fft.ifft(filtered_fft)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Original Signal')
plt.plot(t, filtered_signal.real, label='Filtered Signal')
plt.legend()
plt.grid()
plt.title("Band-pass Filtering using FFT")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
