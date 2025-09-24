# app.py
import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import pandas as pd
import gdown
import zipfile

# ================ CONFIG ================
MODEL_ID_ZIP = "1zh2_UNG3I2etVkzle3_EvrBJ2UGS3if5"  # ID of your zip file (or h5 file renamed .zip)
ZIP_PATH = "waste_model.zip"
MODEL_PATH = "waste_model.h5"

labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ================ DOWNLOAD AND EXTRACT ================
if not os.path.exists(MODEL_PATH):
    # Download the zip (or the file) from Drive
    st.info("Downloading model …")
    url = f"https://drive.google.com/uc?id={MODEL_ID_ZIP}"
    gdown.download(url, ZIP_PATH, quiet=False)
    # Extract h5 file
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
    except zipfile.BadZipFile:
        st.warning("Downloaded file is not a valid ZIP. Maybe it's the raw .h5 already.")
        # Maybe ZIP_PATH is already the h5 file
        if os.path.exists(ZIP_PATH):
            os.rename(ZIP_PATH, MODEL_PATH)

# ================ LOAD MODEL ================
model = keras.models.load_model(MODEL_PATH)

# ================ STREAMLIT UI ================
st.title("Waste Classification App")
st.write("Upload an image to classify:")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((224, 224))
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds_raw = model.predict(arr)[0]
    # Convert to probabilities if needed
    exps = np.exp(preds_raw - np.max(preds_raw))
    probs = exps / exps.sum()

    # Show top-3
    top_k = 3
    top_idx = np.argsort(probs)[::-1][:top_k]

    st.subheader("Top Predictions")
    for i in top_idx:
        st.write(f"{labels[i]} — {probs[i]*100:.2f}%")

    # Show all probabilities as bar chart
    df = pd.DataFrame({"Class": labels, "Probability": probs})
    st.subheader("Probability Distribution")
    st.bar_chart(df.set_index("Class"))




