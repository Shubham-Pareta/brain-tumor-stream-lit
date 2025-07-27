import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("models/brain_tumor_classifier_model.h5")
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# App config
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Brain Tumor MRI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an MRI image and get the tumor prediction in seconds.</p>", unsafe_allow_html=True)
st.markdown("---")

# File upload section
uploaded_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="🖼️ Uploaded Image", use_column_width=True)

    with col2:
        # Load and preprocess image
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        with st.spinner("🔍 Analyzing..."):
            prediction = model.predict(img_array)
            pred_idx = np.argmax(prediction)
            confidence = float(np.max(prediction))

        # Results
        st.success("✅ Prediction complete!")
        st.markdown(f"<h3 style='color:#36d;'>🧾 Prediction: <b>{class_labels[pred_idx]}</b></h3>", unsafe_allow_html=True)
        st.markdown(f"<h5>📊 Confidence: <b>{confidence * 100:.2f}%</b></h5>", unsafe_allow_html=True)

        # Show class probabilities
        st.markdown("<hr><h4>Class Probabilities:</h4>", unsafe_allow_html=True)
        for i, prob in enumerate(prediction[0]):
            st.markdown(f"- **{class_labels[i]}:** {prob * 100:.2f}%")

else:
    st.info("Please upload an MRI image to begin.")

# Footer
st.markdown("---")