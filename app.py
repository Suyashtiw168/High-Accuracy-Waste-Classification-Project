# app.py
import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import pandas as pd
import gdown, zipfile

# Google Drive file ID (zip file ka ID)
MODEL_ID = "1zh2_UNG3I2etVkzle3_EvrBJ2UGS3if5"
ZIP_PATH = "waste_model.zip"
MODEL_PATH = "waste_model.h5"

labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Download & extract model
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model …")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")

# Load model
model = keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.title("Waste Classification App")
uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded", use_column_width=True)

    img_resized = img.resize((224, 224))
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds_raw = model.predict(arr)[0]

    # softmax
    exps = np.exp(preds_raw - np.max(preds_raw))
    probs = exps / exps.sum()

    # show top-3
    top_idx = np.argsort(probs)[::-1][:3]
    for i in top_idx:
        st.write(f"{labels[i]} — {probs[i]*100:.2f}%")

    df = pd.DataFrame({"Class": labels, "Probability": probs})
    st.bar_chart(df.set_index("Class"))



