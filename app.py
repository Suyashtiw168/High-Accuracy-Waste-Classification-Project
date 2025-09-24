# app.py
import os
import zipfile
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
import gdown

# CONFIG - use your zip file ID (you already gave this)
MODEL_ID = "1RpYLaStWEegQPKOashQoSu1ZjAE24Bry"
ZIP_PATH = "waste_model.zip"
MODEL_PATH = "waste_model.h5"

# Change this threshold as needed (0.6 = 60% confident required)
CONFIDENCE_THRESHOLD = 0.60

labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Download & extract (if not present)
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model — please wait...")
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

# Predict
pred_raw = model.predict(arr)
pred_raw = pred_raw.flatten()
probs = to_probs(pred_raw)

top_idx = int(np.argmax(probs))
top_label = labels[top_idx]
top_prob = float(probs[top_idx])

# If high enough confidence -> show single final prediction
if top_prob >= CONFIDENCE_THRESHOLD:
    st.subheader("Final prediction")
    st.write(f"{top_label} — {top_prob*100:.2f}%")
else:
    # uncertain -> show user-friendly message + top-3 suggestions
    st.subheader("Final prediction (uncertain)")
    st.write(f"Model confidence is low ({top_prob*100:.2f}%). Showing suggestions:")
    top_k = min(3, len(labels))
    top_indices = np.argsort(probs)[::-1][:top_k]
    for i in top_indices:
        st.write(f"{labels[int(i)]} — {probs[int(i)]*100:.2f}%")

# Collapseable debug: show all probs (can be hidden)
with st.expander("Show full class probabilities"):
    df = pd.DataFrame({"Class": labels, "Probability": probs})
    df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
    st.table(df.style.format({"Probability":"{:.4f}"}))





