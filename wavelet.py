import numpy as np
import cv2

def haar_wavelet_transform_2d(img):
    """
    Perform a basic 2D Haar wavelet-like transformation.
    Works for even-sized grayscale images.
    """
    rows, cols = img.shape
    # Ensure even dimensions
    rows -= rows % 2
    cols -= cols % 2
    img = img[:rows, :cols]

    # Horizontal transform
    img_H = np.zeros((rows, cols // 2))
    for i in range(rows):
        img_H[i, :] = (img[i, ::2] + img[i, 1::2]) / 2

    # Vertical transform
    img_V = np.zeros((rows // 2, cols // 2))
    for j in range(cols // 2):
        img_V[:, j] = (img_H[::2, j] + img_H[1::2, j]) / 2

    return img_V

def w2d(img, level=1):
    """
    Applies wavelet-like transformation to an image.
    Handles grayscale and RGB images.
    """
    # Check if the image is grayscale or RGB
    if len(img.shape) == 3:  # RGB Image
        channels = cv2.split(img)
        processed_channels = []
        for ch in channels:
            # Process each channel independently
            ch_float = np.float32(ch) / 255.0
            transformed = ch_float
            for _ in range(level):
                transformed = haar_wavelet_transform_2d(transformed)
            processed_channels.append(transformed)
        # Merge the processed channels back
        processed_img = cv2.merge(processed_channels)
    else:  # Grayscale Image
        img_float = np.float32(img) / 255.0
        processed_img = img_float
        for _ in range(level):
            processed_img = haar_wavelet_transform_2d(processed_img)

    # Scale processed image back to 0-255 range
    processed_img = np.abs(processed_img)
    processed_img *= 255
    return np.uint8(processed_img)

# Example Usage
if __name__ == "__main__":
    # Replace 'your_image.jpg' with the path to your image
    img = cv2.imread('your_image.jpg')  # Reads the image in BGR format
    if img is None:
        raise ValueError("Image not found. Please provide a valid image path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    processed_img = w2d(img, level=2)

    # Display the image
    cv2.imshow("Processed Image", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
