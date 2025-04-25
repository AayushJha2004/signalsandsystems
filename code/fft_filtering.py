import numpy as np
import matplotlib.pyplot as plt

# Sampling settings
fs = 500
t = np.arange(0, 2.0, 1/fs)

# Create signal: frequencies inside and outside the band
signal = (
    np.sin(2*np.pi*20*t) +     # outside
    np.sin(2*np.pi*50*t) +     # inside
    np.sin(2*np.pi*75*t) +     # inside
    np.sin(2*np.pi*100*t) +    # inside
    np.sin(2*np.pi*150*t) +    # outside
    np.sin(2*np.pi*200*t)      # outside
)
signal += 0.5 * np.random.randn(len(t))  # Add noise

# FFT
fft_signal = np.fft.fft(signal)
freq = np.fft.fftfreq(len(t), d=1/fs)

# Band-pass filter mask
passband_mask = (np.abs(freq) >= 40) & (np.abs(freq) <= 130)
filtered_fft = np.zeros_like(fft_signal)
filtered_fft[passband_mask] = fft_signal[passband_mask]

# IFFT to get filtered signal
filtered_signal = np.fft.ifft(filtered_fft)

# Frequency axis and magnitude (shifted for better view)
freq_shifted = np.fft.fftshift(freq)
orig_fft_mag = np.abs(np.fft.fftshift(fft_signal))
filt_fft_mag = np.abs(np.fft.fftshift(filtered_fft))

# Plotting
plt.figure(figsize=(12, 6))

# Time domain
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.plot(t, filtered_signal.real, label='Filtered Signal')
plt.title("Time Domain: Original vs Filtered Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Frequency domain (Filtered)
plt.subplot(2, 1, 2)
plt.plot(freq_shifted, filt_fft_mag, label='Filtered FFT Magnitude')

# Highlight band-pass region
plt.axvspan(40, 130, color='orange', alpha=0.3, label='Pass Band (40–130 Hz)')
plt.axvspan(-130, -40, color='orange', alpha=0.3)
plt.title("Frequency Domain: Filtered FFT (Only Pass Band Retained)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
