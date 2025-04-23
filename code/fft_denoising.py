import numpy as np
import matplotlib.pyplot as plt

# Create a noisy signal
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(t.size)

# FFT
freq_domain = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(t), d=t[1] - t[0])

# Apply low-pass filter
cutoff = 100
filtered_freq_domain = freq_domain.copy()
filtered_freq_domain[np.abs(frequencies) > cutoff] = 0

# Inverse FFT
filtered_signal = np.fft.ifft(filtered_freq_domain)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Noisy Signal')
plt.plot(t, filtered_signal.real, label='Filtered Signal')
plt.legend()
plt.title("Signal Denoising using FFT")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
