
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from skimage import data, color

# === Custom FFT and IFFT ===

def fft_recursive(a):
    n = len(a)
    if n == 1:
        return a
    a_even = fft_recursive(a[::2])
    a_odd = fft_recursive(a[1::2])
    factor = np.exp(-2j * np.pi * np.arange(n) / n)
    return np.concatenate([
        a_even + factor[:n // 2] * a_odd,
        a_even - factor[:n // 2] * a_odd
    ])

def ifft_recursive(A):
    n = len(A)
    if n == 1:
        return A
    A_conj = np.conj(A)
    y = fft_recursive(A_conj)
    return np.conj(y) / n

def pad_to_power_of_two(arr):
    n = len(arr)
    size = 1 << (n - 1).bit_length()
    return np.pad(arr, (0, size - n))

def fft2d(matrix):
    row_fft = np.array([fft_recursive(row) for row in matrix])
    col_fft = np.array([fft_recursive(col) for col in row_fft.T])
    return col_fft.T

def ifft2d(matrix):
    row_ifft = np.array([ifft_recursive(row) for row in matrix])
    col_ifft = np.array([ifft_recursive(col) for col in row_ifft.T])
    return col_ifft.T

# === 1. Synthetic 1D Signal ===

fs = 1000
t = np.arange(0, 1, 1/fs)
x = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)
x_padded = pad_to_power_of_two(x)
X = fft_recursive(x_padded)
f = np.fft.fftfreq(len(X), 1/fs)

plt.plot(f[:len(f)//2], np.abs(X[:len(f)//2]))
plt.title("Custom FFT - 1D Synthetic Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(f)|")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 2. 2D FFT on Image ===

img = color.rgb2gray(data.astronaut())[100:356, 100:356]
img = img.astype(np.float32)
rows, cols = img.shape
r2, c2 = 1 << rows.bit_length(), 1 << cols.bit_length()
img_padded = np.pad(img, ((0, r2 - rows), (0, c2 - cols)))

F = fft2d(img_padded)
magnitude = np.log(1 + np.abs(F))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude, cmap='gray')
plt.title("FFT Magnitude Spectrum (Log Scale)")
plt.axis('off')
plt.tight_layout()
plt.show()

# === 3. Time Series Analysis ===

t = np.linspace(0, 10, 1000)
signal = np.sin(2*np.pi*2*t) + 0.5*np.sin(2*np.pi*6*t + np.pi/4) + 0.3*np.random.randn(len(t))
signal_padded = pad_to_power_of_two(signal)
S = fft_recursive(signal_padded)
f_time = np.fft.fftfreq(len(S), t[1] - t[0])

plt.plot(f_time[:len(f_time)//2], np.abs(S[:len(S)//2]))
plt.title("Custom FFT - Time Series Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(f)|")
plt.grid(True)
plt.tight_layout()
plt.show()