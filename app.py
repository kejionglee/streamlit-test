import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from skimage.feature import hog

# Load the model
model = joblib.load("trained_model_SVM.pkl")

st.title("Medical Image Classifier")
st.write("This app classifies uploaded images as **Positive** or **Negative** using an SVM model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def extract_features(img):
    img_resized = cv2.resize(img, (64, 64))
    features, _ = hog(img_resized,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      visualize=True)
    return features

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to OpenCV format
    img_cv = np.array(image)

    # Preprocess and extract features
    processed = preprocess_image(img_cv)
    features = extract_features(processed).reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    label = "Positive" if prediction == 1 else "Negative"

    st.subheader("Prediction:")
    st.markdown(f"### 🩺 {label}")
