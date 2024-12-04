import joblib
import json
import numpy as np
import base64
import cv2

__class_name_to_number = {}
__class_number_to_name = {}
__model = None


def classify_image(image_base64_data, file_path=None):
    """
    Classify an image based on pre-trained model.
    :param image_base64_data: Base64 encoded string of the image.
    :param file_path: Optional file path for image classification.
    :return: Classification result or None if no face is detected.
    """
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    if not imgs:
        # No faces detected
        print("No faces detected in the image.")
        return None

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))

        # Generate edge-detection-based processed image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(gray_img, 100, 200)  # Use Canny edge detection
        scalled_img_edges = cv2.resize(img_edges, (32, 32))

        # Combine raw and processed images
        scalled_raw_img_flat = scalled_raw_img.reshape(-1, 1)  # Flatten raw image
        scalled_img_edges_flat = scalled_img_edges.reshape(-1, 1)  # Flatten processed image

        combined_img = np.vstack((scalled_raw_img_flat, scalled_img_edges_flat))
        len_image_array = scalled_raw_img_flat.shape[0] + scalled_img_edges_flat.shape[0]

        final = combined_img.reshape(1, len_image_array).astype(float)
        prediction = __model.predict(final)[0]
        probability = np.around(__model.predict_proba(final) * 100, 2).tolist()[0]

        result.append({
            'class': class_number_to_name(prediction),
            'class_probability': probability,
            'class_dictionary': __class_name_to_number
        })

    return result


def class_number_to_name(class_num):
    """
    Convert class number to class name.
    :param class_num: Class number.
    :return: Class name.
    """
    return __class_number_to_name.get(class_num, "Unknown")


def load_saved_artifacts():
    """
    Load saved model artifacts, including class dictionary and trained model.
    """
    print("Loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("Loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    """
    Convert base64 image string to OpenCV image.
    :param b64str: Base64 encoded string of the image.
    :return: Decoded OpenCV image or None if decoding fails.
    """
    if ',' in b64str:
        encoded_data = b64str.split(',')[1]
    else:
        encoded_data = b64str

    try:
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding Base64 string: {e}")
        return None


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    """
    Detect faces with at least two eyes in the image.
    :param image_path: File path of the image.
    :param image_base64_data: Base64 encoded image string.
    :return: List of cropped face images.
    """
    face_cascade = cv2.CascadeClassifier(
        './opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        './opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
        print("Error: Image data is invalid.")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces


def get_b64_test_image_for_virat():
    """
    Get a Base64 string for a test image.
    :return: Base64 string of the image.
    """
    with open("./b64.txt") as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()
