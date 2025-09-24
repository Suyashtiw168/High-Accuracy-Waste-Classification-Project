# app.py
import os
import io
import numpy as np
from PIL import Image
import streamlit as st

# TensorFlow/Keras import
from tensorflow import keras

# ============= CONFIG =============
MODEL_PATH = "model.keras"
DRIVE_FILE_ID = "1zh2_UNG3I2etVkzle3_EvrBJ2UGS3if5"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# IMPORTANT: change these labels if your training used a different order.
# This list assumes alphabetical order of folders:
labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ============= UTILITIES =============
def download_from_gdrive(dest_path: str):
    """Download file from Google Drive using gdown, fallback to requests."""
    try:
        import gdown
        st.info("Downloading model with gdown...")
        gdown.download(DRIVE_URL, dest_path, quiet=False)
        return
    except Exception as e:
        st.warning("gdown not available or failed; falling back to requests. "
                   "Make sure 'gdown' is in requirements.txt for faster/more reliable downloads.")
    # fallback
    try:
        import requests
        with requests.get(DRIVE_URL, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as ex:
        st.error(f"Failed to download model: {ex}")
        raise

@st.cache_resource
def load_model_cached():
    """Downloads (if needed) and loads the Keras model. Cached so it won't reload every rerun."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            download_from_gdrive(MODEL_PATH)
    with st.spinner("Loading model..."):
        model = keras.models.load_model(MODEL_PATH)
    return model

def preprocess_image(pil_img: Image.Image, target_size=(224, 224)):
    """Convert to RGB, resize, scale to [0,1], expand dims."""
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

def safe_to_probs(arr):
    """Convert model outputs to a probability vector robustly."""
    arr = np.array(arr, dtype=np.float64).flatten()
    # If already close to a probability distribution, normalize gently
    s = arr.sum()
    if s <= 0 or not np.isfinite(s) or abs(s - 1.0) > 1e-3:
        # apply softmax for logits or unnormalized outputs
        exps = np.exp(arr - np.max(arr))
        probs = exps / exps.sum()
    else:
        probs = arr / s
    return probs

# ============= APP UI =============
def app():
    st.set_page_config(page_title="♻️ Waste Classifier", layout="centered")
    st.title("♻️ Waste Classification")
    st.write("Upload an image and the model will classify it into waste categories.")
    st.markdown("**Model source:** Google Drive (downloaded at runtime).")

    # Load model (cached)
    model = load_model_cached()

    # Show model info (collapsible)
    with st.expander("Model info (debug)"):
        try:
            st.write(f"Model input shape: {model.input_shape}")
            st.write(f"Model output shape: {model.output_shape}")
        except Exception:
            st.write("Could not read model shapes.")

    # Image uploader and camera input
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    camera_file = None
    # Streamlit's camera input (works in browsers that support it)
    try:
        camera_file = st.camera_input("Or take a picture with your camera")
    except Exception:
        camera_file = None

    img_file = uploaded_file if uploaded_file is not None else camera_file

    if img_file is not None:
        img = Image.open(io.BytesIO(img_file.read())).convert("RGB")
        st.image(img, caption="Input image", use_column_width=True)

        # Preprocess
        arr = preprocess_image(img, target_size=(224, 224))

        # Predict
        with st.spinner("Predicting..."):
            preds_raw = model.predict(arr)
        preds = preds_raw.flatten() if preds_raw.ndim > 1 else preds_raw
        probs = safe_to_probs(preds)

        # Display probabilities
        st.subheader("Class probabilities (debug)")
        # build a dataframe-like display
        try:
            import pandas as pd
            df = pd.DataFrame({"label": labels, "probability": [float(x) for x in probs]})
            df = df.sort_values("probability", ascending=False).reset_index(drop=True)
            st.table(df.style.format({"probability": "{:.4f}"}))
        except Exception:
            for i, lab in enumerate(labels):
                st.write(f"{lab}: {probs[i]:.4f}")

        # Top prediction(s)
        top_k = min(3, len(labels))
        top_idx = np.argsort(probs)[::-1][:top_k]
        st.subheader("Prediction")
        for rank, idx in enumerate(top_idx, start=1):
            st.write(f"{rank}. **{labels[int(idx)]}** — {probs[int(idx)]:.3%}")

        # Optional bar chart visualization
        try:
            st.subheader("Probability distribution")
            chart_df = {labels[i]: float(probs[i]) for i in range(len(labels))}
            # st.bar_chart expects dict/list or dataframe; convert to 2-row df for a horizontal feel
            import pandas as pd
            chart = pd.DataFrame.from_dict(chart_df, orient="index", columns=["probability"])
            st.bar_chart(chart)
        except Exception:
            pass

        # Helpful note
        st.info("If all probabilities are very low or one class dominates always, check (1) label order, (2) preprocessing matches training (size, rescale), (3) whether model was trained with different classes/order.")

    else:
        st.write("Upload an image or use the camera to test the model.")
        st.write("Tip: If the model mis-predicts consistently, enable the debug panel to inspect class probabilities.")

    # Footer / troubleshooting
    st.markdown("---")
    st.markdown("**Troubleshooting:**")
    st.markdown(
        "- If predictions are wrong for many images: confirm training label order. "
        "If you used `flow_from_directory`, class order is alphabetical. "
        "Change the `labels` list at top if needed."
    )
    st.markdown("- Ensure `requirements.txt` includes `gdown`, `tensorflow`, `pillow`, `numpy`, `streamlit`.")
    st.markdown("- For faster deploys, consider placing the model in a cloud storage bucket and using direct download link.")

if __name__ == "__main__":
    app()


