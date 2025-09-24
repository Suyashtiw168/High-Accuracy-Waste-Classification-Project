import os
import gdown
import numpy as np
from tensorflow import keras
from PIL import Image
import streamlit as st

# =====================
# CONFIG
# =====================
MODEL_PATH = "model.keras"
# Google Drive share link se file ID nikalo (tumhari link me ye hai: 1zh2_UNG3I2etVkzle3_EvrBJ2UGS3if5)
DRIVE_URL = "https://drive.google.com/uc?id=1zh2_UNG3I2etVkzle3_EvrBJ2UGS3if5"

# =====================
# MODEL LOADER
# =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    model = keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =====================
# PREDICTION FUNCTION
# =====================
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(img):
    arr = preprocess_image(img)
    preds = model.predict(arr)
    return preds

# =====================
# STREAMLIT APP
# =====================
def main():
    st.title("♻️ Waste Classification App")
    st.write("Upload an image to classify into waste categories.")

    uploaded = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        preds = predict(img)
        st.write("🔮 Raw Predictions:", preds)

        # Agar tumhare model me class labels defined hain to unko add karo:
        labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        predicted_class = labels[np.argmax(preds)]
        st.success(f"✅ Predicted Class: **{predicted_class}**")

if __name__ == "__main__":
    main()


