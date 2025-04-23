import numpy as np
import time
import matplotlib.pyplot as plt

# --- FFT Functions (from before) ---

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

# --- Polynomial Multiplication: FFT O(n log n) ---

def fft_poly_mult(a, b):
    size = 1 << ((len(a) + len(b) - 2).bit_length())
    A = np.array(a + [0] * (size - len(a)), dtype=complex)
    B = np.array(b + [0] * (size - len(b)), dtype=complex)
    FA = fft_recursive(A)
    FB = fft_recursive(B)
    FC = FA * FB
    result = ifft_recursive(FC)
    return np.round(result.real, 6).tolist()

# --- Polynomial Multiplication: Naive O(n^2) ---

def naive_poly_mult(a, b):
    m, n = len(a), len(b)
    result = [0] * (m + n - 1)
    for i in range(m):
        for j in range(n):
            result[i + j] += a[i] * b[j]
    return result

# --- Benchmark and Plot ---

ns = [2**i for i in range(3, 11)]  # n = 8 to 1024
times_naive = []
times_fft = []
n2_curve = []
nlogn_curve = []

for n in ns:
    a = np.random.rand(n).tolist()
    b = np.random.rand(n).tolist()

    # Naive
    start = time.time()
    naive_poly_mult(a, b)
    times_naive.append(time.time() - start)

    # FFT
    start = time.time()
    fft_poly_mult(a, b)
    times_fft.append(time.time() - start)

    # Theoretical curves (scaled for comparison)
    n2_curve.append((n**2) / 1e6)
    nlogn_curve.append((n * np.log2(n)) / 1e5)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(ns, times_naive, 'o-', label='Naive O(n²)')
plt.plot(ns, times_fft, 's-', label='FFT O(n log n)')
plt.plot(ns, n2_curve, '--', label='n² (scaled)')
plt.plot(ns, nlogn_curve, '--', label='n log n (scaled)')
plt.xlabel('Polynomial Length (n)')
plt.ylabel('Time (seconds)')
plt.title('Polynomial Multiplication Time Complexity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()