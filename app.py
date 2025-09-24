import streamlit as st
import numpy as np
import requests, os
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# Google Drive से मॉडल डाउनलोड करो (अगर local में नहीं है)
MODEL_URL = "https://drive.google.com/uc?id=1zh2_UNG3I2etVkzle3_EvrBJ2UGS3if5"
MODEL_PATH = "waste_model.keras"

if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL, allow_redirects=True)
    open(MODEL_PATH, 'wb').write(r.content)

# मॉडल load करो
model = keras.models.load_model(MODEL_PATH)

# Labels (confirmed order)
labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

st.title("Waste Classification App")
st.write("Upload an image and the model will classify it into one of 6 categories.")

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

    # Top-3 predictions निकालो
    top_indices = preds.argsort()[-3:][::-1]
    st.subheader("Top Predictions:")
    for i in top_indices:
        st.write(f"{labels[i]} — {preds[i]*100:.2f}%")

    # Probability distribution graph
    st.subheader("Probability distribution")
    fig, ax = plt.subplots()
    ax.bar(labels, preds, color="skyblue")
    plt.xticks(rotation=45)
    st.pyplot(fig)



