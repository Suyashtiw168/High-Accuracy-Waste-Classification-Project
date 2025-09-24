import streamlit as st
import numpy as np
import os
from tensorflow import keras
from PIL import Image
import pandas as pd
import gdown

# Google Drive model file ID
MODEL_ID = "1zh2_UNG3I2etVkzle3_EvrBJ2UGS3if5"  # इसे अपने .h5 फाइल का ID डालना
MODEL_PATH = "waste_model.h5"

# Model download (अगर local में नहीं है तो)
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

# Load model
model = keras.models.load_model(MODEL_PATH)

# Labels (confirmed order)
labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

st.title("Waste Classification App")
st.write("Upload an image and the model will classify it into one of 6 categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Prediction
    preds = model.predict(arr)[0]

    # Top-3 predictions
    top_indices = preds.argsort()[-3:][::-1]
    st.subheader("Top Predictions:")
    for i in top_indices:
        st.write(f"{labels[i]} — {preds[i]*100:.2f}%")

    # Probability distribution (Streamlit chart)
    st.subheader("Probability distribution")
    chart_data = pd.DataFrame({"Class": labels, "Probability": preds})
    st.bar_chart(chart_data.set_index("Class"))




