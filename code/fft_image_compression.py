import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize

# Load and preprocess image
image = color.rgb2gray(data.astronaut())
image = resize(image, (256, 256))

# Apply 2D FFT
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Compress: keep only high-magnitude components
magnitude = np.abs(fshift)
threshold = np.percentile(magnitude, 95)
fshift[magnitude < threshold] = 0

# Inverse FFT
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift).real

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Compressed Image")
plt.imshow(img_back, cmap='gray')
plt.axis('off')
plt.show()
