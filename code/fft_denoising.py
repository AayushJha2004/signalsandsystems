import numpy as np
import matplotlib.pyplot as plt

# Create a noisy signal
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(t.size)

# FFT
freq_domain = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(t), d=t[1] - t[0])

# Apply low-pass filter
cutoff = 100  # Hz
filtered_freq_domain = freq_domain.copy()
filtered_freq_domain[np.abs(frequencies) > cutoff] = 0

# Inverse FFT
filtered_signal = np.fft.ifft(filtered_freq_domain)

# Shifted frequency and magnitude (for better visualization)
freq_shifted = np.fft.fftshift(frequencies)
orig_fft_mag = np.abs(np.fft.fftshift(freq_domain))
filt_fft_mag = np.abs(np.fft.fftshift(filtered_freq_domain))

# Plot
plt.figure(figsize=(12, 6))

# Time domain plot
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Noisy Signal')
plt.plot(t, filtered_signal.real, label='Filtered Signal')
plt.legend()
plt.title("Time Domain: Noisy vs Filtered Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()

# Frequency domain plot
plt.subplot(2, 1, 2)
plt.plot(freq_shifted, filt_fft_mag, label='Filtered FFT Magnitude')

# Highlight pass band
plt.axvspan(-cutoff, cutoff, color='orange', alpha=0.3, label=f'Pass Band (|f| ≤ {cutoff} Hz)')
plt.title("Frequency Domain: Filtered FFT (Low-Pass)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
