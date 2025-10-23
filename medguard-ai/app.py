import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random

# --- CONFIG ---
st.set_page_config(page_title="MedGuard AI", page_icon="💊")

# --- TITLE ---
st.title("💊 MedGuard AI")
st.caption("Verify your medicine — one scan at a time")

# --- Upload or NAFDAC Input ---
uploaded_file = st.file_uploader("📸 Upload drug image", type=["jpg", "png", "jpeg"])
nafdac_number = st.text_input("Or enter NAFDAC number")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model")  # Folder containing the trained Teachable Machine model
    return model

model = load_model()

# --- Mock NAFDAC Data ---
nafdac_data = {"A12345": "Authentic", "B67890": "Fake", "C11223": "Authentic"}

# --- Predict Function ---
def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = round(np.max(predictions) * 100, 2)
    label = "Authentic" if class_index == 0 else "Possibly Fake"
    return label, confidence

# --- Safety Tips ---
tips = [
    "Always check the NAFDAC number before purchase.",
    "Buy drugs only from licensed pharmacies.",
    "Avoid medications with tampered seals or faded prints."
]

# --- Verification Logic ---
if st.button("🔍 Verify Drug"):
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Drug Image", use_column_width=True)
        label, confidence = predict_image(image)
        st.success(f"Result: {label}")
        st.info(f"Confidence Score: {confidence}%")
        st.write("💡 Tip:", random.choice(tips))

    elif nafdac_number:
        result = nafdac_data.get(nafdac_number, "Unverified")
        st.write(f"🔢 NAFDAC Verification Result: **{result}**")
        st.write("💡 Tip:", random.choice(tips))
    else:
        st.warning("Please upload an image or enter a NAFDAC number.")
