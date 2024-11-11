import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\Kaushiki\Downloads\pneumonia_detection_model (2).keras')

st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to check for signs of pneumonia.")

# Upload an image file
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))  # Resize to model's expected input
    img_normalized = img_resized / 255.0  # Normalize
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Expand dims to fit model input

    # Make prediction
    prediction = model.predict(img_expanded)
    confidence = prediction[0][0] * 100  # Calculate confidence score

    # Display result
    if prediction[0] > 0.5:
        st.write(f"**Prediction:** Pneumonia detected with **confidence: {confidence:.2f}%**")
    else:
        st.write(f"**Prediction:** No pneumonia detected with **confidence: {100 - confidence:.2f}%**")
