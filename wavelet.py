import numpy as np
import pywt
import cv2

def w2d(img, mode='haar', level=1):
    '''
    Apply Wavelet Transform to an image.
    '''
    if len(img.shape) == 3:
        # Convert to grayscale if the image is in color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Normalize and convert to float
    imArray = np.float32(img) / 255

    # Compute wavelet coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process coefficients: Zero out approximation coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # Reconstruct the image
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.clip(imArray_H * 255, 0, 255).astype(np.uint8)

    return imArray_H
