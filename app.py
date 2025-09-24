# app.py
import os
import zipfile
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
import gdown

# ---------- CONFIG ----------
# (यह ID तुम्हारे दिए हुए waste_model.zip की ID है)
MODEL_ID = "1RpYLaStWEegQPKOashQoSu1ZjAE24Bry"
ZIP_PATH = "waste_model.zip"
MODEL_PATH = "waste_model.h5"

# Confirmed class order from training
labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ---------- DOWNLOAD & EXTRACT ----------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model — please wait...")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    try:
        gdown.download(url, ZIP_PATH, quiet=False)
    except Exception as e:
        st.error("Model download failed: " + str(e))
        st.stop()

    # Try to extract zip; if it's actually an h5 file, rename it
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(".")
    except zipfile.BadZipFile:
        # If not a zip, maybe it's directly the h5; rename
        if os.path.exists(ZIP_PATH):
            os.rename(ZIP_PATH, MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file {MODEL_PATH} not found after download/extract.")
        st.stop()

# ---------- LOAD MODEL ----------
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error("Failed to load model: " + str(e))
    st.stop()

# ---------- UTILS ----------
def to_probabilities(output_array):
    arr = np.array(output_array).flatten()
    # If already normalized, keep; otherwise apply softmax
    s = arr.sum()
    if s <= 0 or not np.isfinite(s) or abs(s - 1.0) > 1e-3 or np.any(arr < 0):
        exps = np.exp(arr - np.max(arr))
        probs = exps / exps.sum()
    else:
        probs = arr / s
    return probs

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Waste Classifier", layout="centered")
st.title("Waste Classification")
st.write("Upload an image. The app will display a single final prediction (most likely class).")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input image", use_column_width=True)

    # Preprocess according to training input shape (224x224, normalized)
    img_resized = img.resize((224, 224))
    arr = np.asarray(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1,224,224,3)

    # Predict
    preds_raw = model.predict(arr)
    preds_raw = preds_raw.flatten()
    probs = to_probabilities(preds_raw)

    # Top-1 final prediction
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    top_prob = probs[top_idx]

    st.subheader("Final prediction")
    st.write(f"**{top_label}** — {top_prob*100:.2f}%")

    # Optional: show full probabilities in collapsible panel
    with st.expander("Show all class probabilities"):
        df = pd.DataFrame({"Class": labels, "Probability": probs})
        df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
        st.table(df.style.format({"Probability": "{:.4f}"}))

else:
    st.write("Upload an image to get a prediction.")





