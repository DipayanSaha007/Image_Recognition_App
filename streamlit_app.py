import streamlit as st
import numpy as np
import base64
from PIL import Image
import io
import util  # Assuming util contains your classification logic.

# Load the model and artifacts
util.load_saved_artifacts()

# Function to classify image
def classify_image(image_data):
    response = util.classify_image(image_data)
    return response

# Streamlit App
st.title("Sports Person Classifier")

# Sidebar
st.sidebar.header("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Add a section for displaying classifiable sports personalities
st.subheader("Sports Personalities We Can Classify")

# List of sports personalities and their image paths
personalities = [
    {"name": "Lionel Messi", "image_path": "./images/messi.jpeg"},
    {"name": "Maria Sharapova", "image_path": "./images/sharapova.jpeg"},
    {"name": "Roger Federer", "image_path": "./images/federer.jpeg"},
    {"name": "Serena Williams", "image_path": "./images/serena.jpeg"},
    {"name": "Virat Kohli", "image_path": "./images/virat.jpeg"},
]

# Display personalities in a grid
cols = st.columns(5)  # Adjust the number based on your layout preference
for idx, personality in enumerate(personalities):
    with cols[idx % 5]:  # Cycle through columns
        st.image(personality["image_path"], caption=personality["name"], use_container_width=True)

st.write("Use Sidebar to Upload & Classify an image")
# Image classification logic
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert the image to base64 for backend processing
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    if st.sidebar.button("Classify"):
        # Call the classification function
        result = classify_image(img_data)

        if result:
            # Parse result
            st.success("Classification Result:")
            for item in result:
                player = item["class"]
                probabilities = item["class_probability"]
                class_dict = item["class_dictionary"]

                # Display probabilities in a table
                st.write(f"### {player.capitalize()}")
                data = [(k.capitalize(), round(probabilities[v], 2)) for k, v in class_dict.items()]
                st.table(data)
        else:
            st.error("Could not classify the image. Ensure the image contains a clear face.")

