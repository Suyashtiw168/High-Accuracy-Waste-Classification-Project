# app.py
import os
import zipfile
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
import gdown

# CONFIG
MODEL_ID = "1RpYLaStWEegQPKOashQoSu1ZjAE24Bry"   # Google Drive file ID
ZIP_PATH = "waste_model.zip"
MODEL_PATH = "waste_model.h5"

labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Download & extract (if not present)
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model …")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    try:
        gdown.download(url, ZIP_PATH, quiet=False)
    except Exception as e:
        st.error("Model download failed: " + str(e))
        st.stop()

    # Try extract, otherwise rename if raw h5
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(".")
    except zipfile.BadZipFile:
        if os.path.exists(ZIP_PATH):
            os.rename(ZIP_PATH, MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file {MODEL_PATH} not found after download/extract.")
        st.stop()

# Load model
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error("Failed to load model: " + str(e))
    st.stop()

def to_probs(arr):
    a = np.array(arr).flatten()
    # if not normalized, apply softmax
    s = a.sum()
    if s <= 0 or not np.isfinite(s) or abs(s - 1.0) > 1e-3 or np.any(a < 0):
        exps = np.exp(a - np.max(a))
        return exps / exps.sum()
    return a / s

st.set_page_config(page_title="Waste Classifier", layout="centered")
st.title("Waste Classification")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded is None:
    st.write("Upload an image to get prediction.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Input image", use_column_width=True)

# Preprocess: resize to 224x224 and scale (match your training)
img_resized = img.resize((224, 224))
arr = np.asarray(img_resized).astype("float32") / 255.0
arr = np.expand_dims(arr, axis=0)

# Predict (single image, no steps, silent)
pred_raw = model.predict(arr, verbose=0)
probs = to_probs(pred_raw)

top_idx = int(np.argmax(probs))
top_label = labels[top_idx]
top_prob = float(probs[top_idx])

# Final result only
st.subheader("Final Prediction")
st.write(f"{top_label} — {top_prob*100:.2f}%")






