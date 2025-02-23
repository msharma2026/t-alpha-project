import streamlit as st
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@st.cache_resource
def load_emotion_model():
    model_path = "emotion_model.h5"
    if not os.path.exists(model_path):
        st.error("Model file 'emotion_model.h5' not found in the current directory.")
        st.stop()
    return load_model(model_path)

# Load the model from the local file
model = load_emotion_model()
st.write("Model loaded successfully!")

# Choose Input Mode: Upload an Image or Take a Picture
option = st.radio("Choose input method:", ("Upload an Image", "Take a Picture"))
image = None  # Placeholder for image

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
elif option == "Take a Picture":
    captured_image = st.camera_input("Take a picture")
    if captured_image is not None:
        file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

# Process and predict emotion if an image is provided
if image is not None:
    # Resize to 48x48 (model's expected input size)
    resized = cv2.resize(image, (48, 48))
    
    # Ensure RGB format (3 channels)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize and expand dimensions
    processed_img = img_to_array(rgb_image) / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)

    # Predict emotion
    preds = model.predict(processed_img)
    emotion = EMOTIONS[np.argmax(preds)]
    
    # Display the image and prediction using use_container_width
    st.image(image, caption="Captured Image", use_container_width=True)
    st.write(f"Predicted Emotion: **{emotion}**")
