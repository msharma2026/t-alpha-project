import streamlit as st
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import PyPDF2
from PIL import Image
import tempfile

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@st.cache_resource
def load_emotion_model():
    model_path = "emotion_model.h5"
    if not os.path.exists(model_path):
        st.error("Model file 'emotion_model.h5' not found in the current directory.")
        st.stop()
    return load_model(model_path)

model = load_emotion_model()
st.write("Model loaded successfully!")

tabp = st.sidebar.radio("Choose tab: ", ["Empathy Check", "Model Confidence"])

if tabp == "Model Confidence":
    st.title("Model")
    st.write("Checks the model's confidence in predicting the emotion of a given face.")
    st.write("")
    st.write("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30), maxSize=(1000, 1000))
        
        if len(faces) == 0:
            st.warning("No face detected!")
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_rgb = np.stack((roi_gray,) * 3, axis=-1)
                roi = roi_rgb.astype("float") / 255.0
                roi = np.expand_dims(roi, axis=0)
                
                preds = model.predict(roi)
                confidence = np.max(preds)
                emotion = EMOTIONS[np.argmax(preds)]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.info(f"Emotion: {emotion} (Confidence: {confidence:.2%})")

if tabp == "Empathy Check":
    st.title("Empathy Check System")
    st.write("This app analyzes facial expressions to assess empathy towards displayed content.")

    #Tabs for different content types
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Upload Video", "Read Content"])
    uploaded_content = None
    expected_emotion = ""

    with tab1:
        st.header("Upload an Image")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        uploaded_content = uploaded_image

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with tab2:
        st.header("Upload a Video")
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
        uploaded_content = uploaded_video

        if uploaded_video is not None:
            st.video(uploaded_video)

    with tab3:
        st.header("Read Content")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        displayed_text = ""
        
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                displayed_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            else:
                displayed_text = "Error"
                
        uploaded_content = uploaded_file
        st.text_area("Content:", displayed_text, height=500)

    #Capture video feed
    cap = cv2.VideoCapture(0)
    emotion = ""

    #Expanded expected emotions dictionary with grouped emotions
    expected_emotions = {
        "default": [],
        "sad": ["Sad", "Fear"],
        "happy": ["Happy", "Surprise"],
        "distressing": ["Fear", "Angry"],
        "neutral": ["Neutral"],
        "angry": ["Angry", "Disgust"],
        "disgusted": ["Disgust", "Angry"],
        "surprised": ["Surprise", "Happy"]
    }

    content_type = st.radio("Select Content Type", list(expected_emotions.keys()))
    if content_type != "default":
        expected_emotion = expected_emotions[content_type]

    emotionp = st.empty()
    empathyp = st.empty()

    if uploaded_content is not None:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi_rgb = np.stack((roi_gray,) * 3, axis=-1)
                    roi = roi_rgb.astype("float") / 255.0
                    roi = np.expand_dims(roi, axis=0)

                    preds = model.predict(roi)[0]
                    emotion = EMOTIONS[np.argmax(preds)]

                    emotionp.write(f"Detected Emotion: {emotion}")

                    if emotion in expected_emotion:
                        empathyp.success("Empathy detected!")
                    else:
                        empathyp.warning("No Empathy detected.")
