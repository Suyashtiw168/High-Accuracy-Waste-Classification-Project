import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load model (.keras format)
model = load_model("high_accuracy_waste_classification_latest.keras")

# Class labels (adjust if needed)
class_labels = ["plastic", "metal", "cardboard", "trash", "paper", "glass"]

st.title("🗑️ Waste Classification App (ResNet50)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)
    pred_index = np.argmax(prediction)
    pred_label = class_labels[pred_index]
    confidence = float(np.max(prediction))

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.success(f"Prediction: {pred_label} ({confidence*100:.2f}% confidence)")

