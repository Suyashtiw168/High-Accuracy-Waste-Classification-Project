import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import pandas as pd
import gdown
import zipfile

# === CONFIG ===
MODEL_ID = "1RpYLaStWEegQPKOashQoSu1ZjAE24Bry"  # zip file ID
ZIP_PATH = "waste_model.zip"
MODEL_PATH = "waste_model.h5"

labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# === DOWNLOAD & EXTRACT ===
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model …")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)

    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(".")
    except zipfile.BadZipFile:
        st.warning("The downloaded file is not a valid zip. Maybe it’s already the .h5 file.")
        if os.path.exists(ZIP_PATH):
            os.rename(ZIP_PATH, MODEL_PATH)

# === LOAD MODEL ===
model = keras.models.load_model(MODEL_PATH)

# === UI ===
st.title("Waste Classification App")
st.write("Upload an image to classify it into waste categories.")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((224, 224))
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds_raw = model.predict(arr)[0]

    # Softmax
    exps = np.exp(preds_raw - np.max(preds_raw))
    probs = exps / exps.sum()

    # Top-3
    top_idx = np.argsort(probs)[::-1][:3]
    st.subheader("Top Predictions")
    for i in top_idx:
        st.write(f"{labels[i]} — {probs[i]*100:.2f}%")

    df = pd.DataFrame({"Class": labels, "Probability": probs})
    st.subheader("Probability Distribution")
    st.bar_chart(df.set_index("Class"))




