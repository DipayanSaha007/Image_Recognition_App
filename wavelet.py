import numpy as np
import cv2

def haar_wavelet_transform_2d(img):
    '''
    Apply the 2D Haar wavelet transform to an image.
    This is a very simplified Haar transform.
    '''
    # Ensure the image is in float and normalized
    img = np.float32(img) / 255.0

    # Image dimensions
    rows, cols = img.shape

    # Step 1: Horizontal wavelet transform
    img_H = np.zeros_like(img)
    for i in range(rows):
        # If the number of columns is odd, adjust slices to ensure same length
        if cols % 2 != 0:
            img_H[i, :-1:2] = (img[i, :-1:2] + img[i, 1::2]) / 2  # Skip last column for odd-sized rows
        else:
            img_H[i] = (img[i, ::2] + img[i, 1::2]) / 2

    # Step 2: Vertical wavelet transform
    img_V = np.zeros_like(img_H)
    for j in range(cols):
        # If the number of rows is odd, adjust slices to ensure same length
        if rows % 2 != 0:
            img_V[:-1:2, j] = (img_H[:-1:2, j] + img_H[1::2, j]) / 2  # Skip last row for odd-sized columns
        else:
            img_V[:, j] = (img_H[::2, j] + img_H[1::2, j]) / 2

    return img_V

def w2d(img, mode='haar', level=1):
    '''
    Apply Wavelet Transform to an image (without PyWavelets).
    This implementation only supports Haar wavelet (simplified version).
    '''
    if len(img.shape) == 3:
        # Convert to grayscale if the image is in color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Normalize and convert to float
    imArray = np.float32(img) / 255

    # Apply Haar wavelet transform
    for _ in range(level):
        # For each level, apply 2D Haar transform (simplified version)
        imArray = haar_wavelet_transform_2d(imArray)

    # Zero out the approximation (low-frequency) coefficients
    imArray[imArray < 0.5] = 0

    # Scale back and clip to valid range
    imArray_H = np.clip(imArray * 255, 0, 255).astype(np.uint8)

    return imArray_H
