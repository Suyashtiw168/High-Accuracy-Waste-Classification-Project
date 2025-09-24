import streamlit as st
import numpy as np
import os
from tensorflow import keras
from PIL import Image
import pandas as pd
import gdown
import zipfile

# Google Drive model file ID (zip file का ID दो)
MODEL_ID = "PUT_YOUR_ZIP_FILE_ID_HERE"
ZIP_PATH = "waste_model.zip"
MODEL_PATH = "waste_model.h5"

if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", ZIP_PATH, quiet=False)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")  # waste_model.h5 extract होगा

# Model load
model = keras.models.load_model(MODEL_PATH)

# Labels
labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

st.title("Waste Classification App")
st.write("Upload an image and the model will classify it into one of 6 categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]

    top_indices = preds.argsort()[-3:][::-1]
    st.subheader("Top Predictions:")
    for i in top_indices:
        st.write(f"{labels[i]} — {preds[i]*100:.2f}%")

    st.subheader("Probability distribution")
    chart_data = pd.DataFrame({"Class": labels, "Probability": preds})
    st.bar_chart(chart_data.set_index("Class"))




