# -*- coding: utf-8 -*-

!pip install opencv-python scikit-image matplotlib numpy scipy

!pip install ipywidgets

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy import ndimage

image = data.camera()

plt.imshow(image, cmap='gray')
plt.axis('off')

#Negative Transformation

img = data.camera()   # sample image
transformed = 255 - img
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(transformed, cmap='gray')
plt.title("Negative Image")
plt.axis('off')

plt.show()

#Gamma Transformation

import ipywidgets as widgets
from IPython.display import display

img = data.moon()
img_float = img / 255.0
def gamma_transform(gamma=1.0, c=1.0):
    transformed = c * (img_float ** gamma)
    transformed = np.clip(transformed, 0, 1)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(transformed, cmap='gray')
    plt.title(f"Gamma={gamma}, c={c}")
    plt.axis('off')

    plt.show()

widgets.interact(
    gamma_transform,
    gamma=widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0),
    c=widgets.FloatSlider(min=0.1, max=10.0, step=0.1, value=1.0)
);

#Log Transformation

img = data.camera()
def show_log_spectrum(c=1.0):

    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)

    magnitude_spectrum = np.abs(f_shift)

    log_spectrum = c * np.log(1 + magnitude_spectrum)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Without Log")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(log_spectrum, cmap='gray')
    plt.title(f"Log Transform (c={c})")
    plt.axis('off')

    plt.show()


widgets.interact(
    show_log_spectrum,
    c=widgets.FloatSlider(min=0.1, max=5.0, step=0.1, value=1.0)
);

#Contrast Stretching

img = data.camera()
def contrast_stretch(factor=0.5, brightness=50):

    # Create low contrast image
    low_contrast = img * factor + brightness
    low_contrast = np.clip(low_contrast, 0, 255)

    arr = low_contrast.astype(float)

    min_r = np.min(arr)
    max_r = np.max(arr)

    stretched = (arr - min_r) / (max_r - min_r)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(low_contrast, cmap='gray')
    plt.title("Low Contrast")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(stretched, cmap='gray')
    plt.title("Contrast Stretched")
    plt.axis('off')

    plt.show()


widgets.interact(
    contrast_stretch,
    factor=widgets.FloatSlider(min=0.1, max=1.0, step=0.1, value=0.5),
    brightness=widgets.IntSlider(min=0, max=100, step=5, value=50)
);

#Thresholding

img = data.text()
def thresholding(threshold):

    binary = img > threshold

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(binary, cmap='gray')
    plt.title(f"Threshold = {threshold}")
    plt.axis('off')

    plt.show()

widgets.interact(
    thresholding,
    threshold=widgets.IntSlider(min=0, max=255, step=1, value=120)
);

#Piecewise Linear Stretching

def piecewise_linear(r, r1, s1, r2, s2):

    if r < r1 and r1 > 0.0:
        return (s1 / r1) * r

    elif r < r2:
        return ((s2 - s1) / (r2 - r1)) * (r - r1) + s1

    elif r2 < 1.0:
        return ((1 - s2) / (1 - r2)) * (r - r2) + s2

    else:
        return s2

img = data.brick()
img = img / 255.0
def update(r1=0.3, s1=0.2, r2=0.7, s2=0.9):

    vectorized = np.vectorize(piecewise_linear)
    transformed = vectorized(img, r1, s1, r2, s2)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(transformed, cmap='gray')
    plt.title("Piecewise Linear Stretch")
    plt.axis('off')

    plt.show()


widgets.interact(
    update,
    r1=widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.3),
    s1=widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.2),
    r2=widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.7),
    s2=widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.9)
);

#Histogram Equalization

img = data.moon()
img = img / 255.0
nbins = 256

hist, bins = np.histogram(img.flatten(), nbins, [0,1])
pdf = hist / np.sum(hist)
cdf = np.cumsum(pdf)
equalized = np.interp(img.flatten(), bins[:-1], cdf)
equalized = equalized.reshape(img.shape)
hist_eq, bins_eq = np.histogram(equalized.flatten(), nbins, [0,1])
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.bar(bins[:-1], hist, width=0.004, color='gray')
plt.title("Original Histogram")

plt.subplot(2,2,3)
plt.imshow(equalized, cmap='gray')
plt.title("Equalized Image")
plt.axis('off')

plt.subplot(2,2,4)
plt.bar(bins_eq[:-1], hist_eq, width=0.004, color='gray')
plt.title("Equalized Histogram")

plt.show()

#Histogram Stretching

import numpy as np
import matplotlib.pyplot as plt
from skimage import data

img = data.moon()
img = img / 255.0

arr_vals = img.flatten()

min_val = np.min(arr_vals)
max_val = np.max(arr_vals)

stretched = (img - min_val) / (max_val - min_val)

hist, bins = np.histogram(img.flatten(), 256, [0,1])
hist_eq, bins_eq = np.histogram(stretched.flatten(), 256, [0,1])

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2,2,2)
plt.bar(bins[:-1], hist, width=0.004, color='gray')
plt.title("Original Histogram")

plt.subplot(2,2,3)
plt.imshow(stretched, cmap='gray')
plt.title("Stretched")
plt.axis('off')

plt.subplot(2,2,4)
plt.bar(bins_eq[:-1], hist_eq, width=0.004, color='gray')
plt.title("Stretched Histogram")

plt.show()

def correlation(img, kernel):

    arr = img.astype(float)
    rows, cols = arr.shape
    kr, kc = kernel.shape

    pr = kr // 2
    pc = kc // 2

    padded = np.zeros((rows + 2*pr, cols + 2*pc))
    padded[pr:pr+rows, pc:pc+cols] = arr

    output = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):

            region = padded[i:i+kr, j:j+kc]
            output[i,j] = np.sum(region * kernel)

    return output

def convolution(img, kernel):

    flipped_kernel = np.flipud(np.fliplr(kernel))

    return correlation(img, flipped_kernel)

kernel = np.ones((3,3)) / 9
result = convolution(img, kernel)

plt.imshow(result, cmap='gray')
plt.title("Filtered Image")
plt.axis('off')

#Convolution and Correlation

img = data.camera()
img = img / 255.0
kernel = np.array([
    [1,1,1],
    [1,-8,1],
    [1,1,1]
]) * -1
def correlation(img, kernel):

    arr = img.astype(float)
    rows, cols = arr.shape
    kr, kc = kernel.shape

    pr = kr // 2
    pc = kc // 2

    padded = np.zeros((rows + 2*pr, cols + 2*pc))
    padded[pr:pr+rows, pc:pc+cols] = arr

    output = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):

            region = padded[i:i+kr, j:j+kc]
            output[i,j] = np.sum(region * kernel)

    return output

def convolution(img, kernel):

    flipped_kernel = np.flipud(np.fliplr(kernel))

    return correlation(img, flipped_kernel)

convolution_output = convolution(img, kernel)
correlation_output = correlation(img, kernel)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(convolution_output, cmap='gray')
plt.title("Convolution")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(correlation_output, cmap='gray')
plt.title("Correlation")
plt.axis('off')

plt.show()

#Smoothening Kernels

from scipy.ndimage import uniform_filter, gaussian_filter

img = data.camera()
img = img / 255.0
n = 5      # box kernel size
sigma = 1  # gaussian sigma

box_smoothened = uniform_filter(img, size=n)
gaussian_smoothened = gaussian_filter(img, sigma=sigma)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(box_smoothened, cmap='gray')
plt.title("Box Smoothened")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(gaussian_smoothened, cmap='gray')
plt.title("Gaussian Smoothened")
plt.axis('off')

plt.show()

#Order Static Median Filtering

def median_filter(img, kernel_size):

    arr = img.astype(float)
    rows, cols = arr.shape

    pr = kernel_size // 2

    padded = np.zeros((rows + 2*pr, cols + 2*pr))
    padded[pr:pr+rows, pr:pr+cols] = arr

    output = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):

            region = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.median(region)

    return output

img = data.text()
img = img / 255.0

filtered = median_filter(img, 3)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(filtered, cmap='gray')
plt.title("Median Filtered")
plt.axis('off')

plt.show()

#Max Filter

import numpy as np

def max_filter_manual(img, kernel_size):

    arr = img.astype(float)
    rows, cols = arr.shape

    pr = kernel_size // 2

    padded = np.zeros((rows + 2*pr, cols + 2*pr))
    padded[pr:pr+rows, pr:pr+cols] = arr

    output = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):

            region = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.max(region)

    return output

img = data.camera()
img = img / 255.0

def update(kernel_size):

    filtered = max_filter_manual(img, kernel_size)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(filtered, cmap='gray')
    plt.title(f"Max Filter (k={kernel_size})")
    plt.axis('off')

    plt.show()


widgets.interact(
    update,
    kernel_size=widgets.IntSlider(min=1, max=9, step=2, value=3)
);

#Min Filtering

def min_filter_manual(img, kernel_size):

    arr = img.astype(float)

    # ensure grayscale
    if len(arr.shape) == 3:
        from skimage.color import rgb2gray
        arr = rgb2gray(arr)

    rows, cols = arr.shape

    pr = kernel_size // 2

    padded = np.zeros((rows + 2*pr, cols + 2*pr))
    padded[pr:pr+rows, pr:pr+cols] = arr

    output = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):

            region = padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.min(region)

    return output

img = data.clock()
img = img / 255.0

def update(kernel_size):

    filtered_img = min_filter_manual(img, kernel_size)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(filtered_img, cmap='gray')
    plt.title(f"Min Filter (k={kernel_size})")
    plt.axis('off')

    plt.show()


widgets.interact(
    update,
    kernel_size=widgets.IntSlider(min=1, max=9, step=2, value=3)
);

#Laplacian Filtering

def convolution(img, kernel):

    arr = img.astype(float)
    rows, cols = arr.shape
    kr, kc = kernel.shape

    pr = kr // 2
    pc = kc // 2

    padded = np.zeros((rows + 2*pr, cols + 2*pc))
    padded[pr:pr+rows, pc:pc+cols] = arr

    output = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):

            region = padded[i:i+kr, j:j+kc]
            output[i,j] = np.sum(region * kernel)

    return output

img = data.moon()
img = img / 255.0

def update(kernel_type):

    if kernel_type == "4-connectivity":
        laplacian_kernel = np.array([
            [0,1,0],
            [1,-4,1],
            [0,1,0]
        ])
    else:
        laplacian_kernel = np.array([
            [1,1,1],
            [1,-8,1],
            [1,1,1]
        ])

    mask = convolution(img, laplacian_kernel)

    sharpened = img - mask

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(sharpened, cmap='gray')
    plt.title(f"Laplacian Sharpened ({kernel_type})")
    plt.axis('off')

    plt.show()


widgets.interact(
    update,
    kernel_type=["4-connectivity","8-connectivity"]
);

#First Order Derivative filtering (Sobel filter)

def correlation(img, kernel):

    arr = img.astype(float)
    rows, cols = arr.shape
    kr, kc = kernel.shape

    pr = kr // 2
    pc = kc // 2

    padded = np.zeros((rows + 2*pr, cols + 2*pc))
    padded[pr:pr+rows, pc:pc+cols] = arr

    output = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):

            region = padded[i:i+kr, j:j+kc]
            output[i,j] = np.sum(region * kernel)

    return output

img = data.camera()
img = img / 255.0

sobel_x = np.array([
    [-1,-2,-1],
    [ 0, 0, 0],
    [ 1, 2, 1]
])

sobel_y = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])

gx = correlation(img, sobel_x)
gy = correlation(img, sobel_y)

magnitude = np.sqrt(gx**2 + gy**2)
magnitude = magnitude / np.max(magnitude)

gx_img = np.clip(gx / np.max(np.abs(gx)), 0, 1)
gy_img = np.clip(gy / np.max(np.abs(gy)), 0, 1)
mag_img = magnitude

sharpened_img = img + mag_img

plt.figure(figsize=(10,10))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(gx_img, cmap='gray')
plt.title("Gx (Vertical edges)")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(gy_img, cmap='gray')
plt.title("Gy (Horizontal edges)")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(mag_img, cmap='gray')
plt.title("Magnitude")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(sharpened_img, cmap='gray')
plt.title("Sharpened")
plt.axis('off')

plt.show()

#DFT calculation

img = data.moon()
img = img / 255.0

F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)

spectrum = np.log(1 + np.abs(F_shifted))
spectrum = spectrum / np.max(spectrum)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(spectrum, cmap='gray')
plt.title("Spectrum")
plt.axis('off')

plt.show()

def freq_filter(img, mask):

    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)

    F_filtered = F_shifted * mask

    result = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    result = np.real(result)

    result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result

def spectrum_func(img):

    F = np.fft.fftshift(np.fft.fft2(img))

    s = np.log(1 + np.abs(F))
    s = (s - np.min(s)) / (np.max(s) - np.min(s))

    return s

rows, cols = img.shape
mask = np.zeros((rows, cols))

center_r, center_c = rows//2, cols//2
radius = 30

for i in range(rows):
    for j in range(cols):
        if (i-center_r)**2 + (j-center_c)**2 < radius**2:
            mask[i,j] = 1

filtered = freq_filter(img, mask)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(filtered, cmap='gray')
plt.title("Low-pass filtered")

plt.show()

#ILPF mask

def spectrum(img):
    F = np.fft.fftshift(np.fft.fft2(img))
    s = np.log(1 + np.abs(F))
    s = (s - np.min(s)) / (np.max(s) - np.min(s))
    return s

def freq_filter(img, mask):

    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)

    F_filtered = F_shifted * mask

    result = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    result = np.real(result)

    result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result

def ilpf_mask(rows, cols, cutoff):

    mask = np.zeros((rows, cols))

    cx = rows // 2
    cy = cols // 2

    for i in range(rows):
        for j in range(cols):

            if np.sqrt((i-cx)**2 + (j-cy)**2) <= cutoff:
                mask[i,j] = 1

    return mask

img = data.camera()
img = img / 255.0
rows, cols = img.shape

def update(cutoff):

    mask = ilpf_mask(rows, cols, cutoff)

    filtered = freq_filter(img, mask)

    s_orig = spectrum(img)
    s_filtered = spectrum(filtered)

    plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(filtered, cmap='gray')
    plt.title("Filtered")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(s_orig, cmap='gray')
    plt.title("Original Spectrum")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(s_filtered, cmap='gray')
    plt.title("Filtered Spectrum")
    plt.axis('off')

    plt.show()


widgets.interact(
    update,
    cutoff=widgets.IntSlider(min=10, max=255, step=5, value=50)
);

#Gaussian LPF mask

def spectrum(img):

    F = np.fft.fftshift(np.fft.fft2(img))

    s = np.log(1 + np.abs(F))
    s = (s - np.min(s)) / (np.max(s) - np.min(s))

    return s

def freq_filter(img, mask):

    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)

    F_filtered = F_shifted * mask

    result = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    result = np.real(result)

    result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result

def glpf_mask(rows, cols, cutoff):

    mask = np.zeros((rows, cols))

    cx = rows // 2
    cy = cols // 2

    for i in range(rows):
        for j in range(cols):

            d = np.sqrt((i - cx)**2 + (j - cy)**2)

            exponent = (d**2) / (2*(cutoff**2))

            mask[i, j] = np.exp(-exponent)

    return mask

img = data.camera()
img = img / 255.0

rows, cols = img.shape

def update(cutoff):

    mask = glpf_mask(rows, cols, cutoff)

    filtered = freq_filter(img, mask)

    s_orig = spectrum(img)
    s_filtered = spectrum(filtered)

    plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(filtered, cmap='gray')
    plt.title("Filtered")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(s_orig, cmap='gray')
    plt.title("Original Spectrum")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(s_filtered, cmap='gray')
    plt.title("Filtered Spectrum")
    plt.axis('off')

    plt.show()


widgets.interact(
    update,
    cutoff=widgets.IntSlider(min=10, max=255, step=5, value=50)
);

#Butterworth LPF

def spectrum(img):

    F = np.fft.fftshift(np.fft.fft2(img))

    s = np.log(1 + np.abs(F))
    s = (s - np.min(s)) / (np.max(s) - np.min(s))

    return s

def freq_filter(img, mask):

    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)

    F_filtered = F_shifted * mask

    result = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    result = np.real(result)

    result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result

def blpf_mask(rows, cols, cutoff, n):

    mask = np.zeros((rows, cols))

    cx = rows // 2
    cy = cols // 2

    for i in range(rows):
        for j in range(cols):

            d = np.sqrt((i-cx)**2 + (j-cy)**2)

            denominator = 1 + (d/cutoff)**(2*n)

            mask[i, j] = 1 / denominator

    return mask

img = data.camera()
img = img / 255.0

rows, cols = img.shape

def update(cutoff, n):

    mask = blpf_mask(rows, cols, cutoff, n)

    filtered = freq_filter(img, mask)

    s_orig = spectrum(img)
    s_filtered = spectrum(filtered)

    plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(filtered, cmap='gray')
    plt.title("Filtered")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(s_orig, cmap='gray')
    plt.title("Original Spectrum")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(s_filtered, cmap='gray')
    plt.title("Filtered Spectrum")
    plt.axis('off')

    plt.show()


widgets.interact(
    update,
    cutoff=widgets.IntSlider(min=10, max=255, step=5, value=50),
    n=widgets.IntSlider(min=1, max=10, step=1, value=2)
);

#Ideal high pass filter

def spectrum(img):

    F = np.fft.fftshift(np.fft.fft2(img))

    s = np.log(1 + np.abs(F))
    s = (s - np.min(s)) / (np.max(s) - np.min(s))

    return s

def freq_filter(img, mask):

    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)

    F_filtered = F_shifted * mask

    result = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    result = np.real(result)

    result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result

def ilpf_mask(rows, cols, cutoff):

    mask = np.zeros((rows, cols))

    cx = rows // 2
    cy = cols // 2

    for i in range(rows):
        for j in range(cols):

            if np.sqrt((i-cx)**2 + (j-cy)**2) <= cutoff:
                mask[i,j] = 1

    return mask

def ihpf_mask(rows, cols, cutoff):

    mask = ilpf_mask(rows, cols, cutoff)

    mask = 1 - mask

    return mask

img = data.camera()
img = img / 255.0

rows, cols = img.shape

def update(cutoff):

    mask = ihpf_mask(rows, cols, cutoff)

    filtered = freq_filter(img, mask)

    s_orig = spectrum(img)
    s_filtered = spectrum(filtered)

    plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(filtered, cmap='gray')
    plt.title("High-pass Filtered")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(s_orig, cmap='gray')
    plt.title("Original Spectrum")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(s_filtered, cmap='gray')
    plt.title("Filtered Spectrum")
    plt.axis('off')

    plt.show()


widgets.interact(
    update,
    cutoff=widgets.IntSlider(min=10, max=255, step=5, value=30)
);

#Gaussian High Pass filter

def spectrum(img):

    F = np.fft.fftshift(np.fft.fft2(img))

    s = np.log(1 + np.abs(F))
    s = (s - np.min(s)) / (np.max(s) - np.min(s))

    return s

def freq_filter(img, mask):

    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)

    F_filtered = F_shifted * mask

    result = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    result = np.real(result)

    result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result

def glpf_mask(rows, cols, cutoff):

    mask = np.zeros((rows, cols))

    cx = rows // 2
    cy = cols // 2

    for i in range(rows):
        for j in range(cols):

            d = np.sqrt((i-cx)**2 + (j-cy)**2)

            exponent = (d**2) / (2*(cutoff**2))

            mask[i,j] = np.exp(-exponent)

    return mask

def ghpf_mask(rows, cols, cutoff):

    mask = glpf_mask(rows, cols, cutoff)

    mask = 1 - mask

    return mask

img = data.camera()
img = img / 255.0

rows, cols = img.shape

def update(cutoff):

    mask = ghpf_mask(rows, cols, cutoff)

    filtered = freq_filter(img, mask)

    s_orig = spectrum(img)
    s_filtered = spectrum(filtered)

    plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(filtered, cmap='gray')
    plt.title("Gaussian High-pass")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(s_orig, cmap='gray')
    plt.title("Original Spectrum")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(s_filtered, cmap='gray')
    plt.title("Filtered Spectrum")
    plt.axis('off')

    plt.show()


widgets.interact(
    update,
    cutoff=widgets.IntSlider(min=10, max=255, step=5, value=30)
);

#Butterworth high pass filter

def spectrum(img):

    F = np.fft.fftshift(np.fft.fft2(img))

    s = np.log(1 + np.abs(F))
    s = (s - np.min(s)) / (np.max(s) - np.min(s))

    return s

def freq_filter(img, mask):

    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)

    F_filtered = F_shifted * mask

    result = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    result = np.real(result)

    result = (result - np.min(result)) / (np.max(result) - np.min(result))

    return result

def blpf_mask(rows, cols, cutoff, n):

    mask = np.zeros((rows, cols))

    cx = rows // 2
    cy = cols // 2

    for i in range(rows):
        for j in range(cols):

            d = np.sqrt((i-cx)**2 + (j-cy)**2)

            mask[i, j] = 1 / (1 + (d/cutoff)**(2*n))

    return mask

def bhpf_mask(rows, cols, cutoff, n):

    mask = blpf_mask(rows, cols, cutoff, n)

    mask = 1 - mask

    return mask

img = data.camera()
img = img / 255.0

rows, cols = img.shape

def update(cutoff, n):

    mask = bhpf_mask(rows, cols, cutoff, n)

    filtered = freq_filter(img, mask)

    s_orig = spectrum(img)
    s_filtered = spectrum(filtered)

    plt.figure(figsize=(10,8))

    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(filtered, cmap='gray')
    plt.title("Butterworth High-pass")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(s_orig, cmap='gray')
    plt.title("Original Spectrum")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(s_filtered, cmap='gray')
    plt.title("Filtered Spectrum")
    plt.axis('off')

    plt.show()


widgets.interact(
    update,
    cutoff=widgets.IntSlider(min=10, max=255, step=5, value=30),
    n=widgets.IntSlider(min=1, max=10, step=1, value=2)
);

#DFT manual implementation

def dft_zeropad(arr):

    M, N = arr.shape

    padded = np.zeros((2*M, 2*N))

    padded[0:M, 0:N] = arr

    return padded

def dft(arr):

    M, N = arr.shape

    F = np.zeros((M,N), dtype=complex)

    for u in range(M):
        for v in range(N):

            acc = 0+0j

            for x in range(M):
                for y in range(N):

                    theta = 2*np.pi*((u*x/M) + (v*y/N))

                    acc += arr[x,y] * np.exp(-1j*theta)

            F[u,v] = acc

    return F

def dftshift(F):

    M, N = F.shape

    shifted = np.zeros((M,N), dtype=complex)

    for u in range(M):
        for v in range(N):

            shifted[u,v] = F[u,v] * (-1)**(u+v)

    return shifted

img = data.camera()
img = img / 255.0

# reduce size because manual DFT is very slow
img = img[0:32,0:32]
padded = dft_zeropad(img)
F = dft(padded)

F_shifted = dftshift(F)

s = np.log(1 + np.abs(F_shifted))

s = (s - np.min(s)) / (np.max(s) - np.min(s))

plt.imshow(s, cmap='gray')
plt.title("DFT Spectrum")
plt.axis('off')
