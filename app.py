import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
# ------------------------------
# Load trained model
# ------------------------------
model = load_model("model_vg1.keras", compile=False)
# Build model graph
model(np.zeros((1,128,128,3)))

# ------------------------------
# Load model metrics
# ------------------------------
with open("metrics.json") as f:
    metrics = json.load(f)

# ------------------------------
# Streamlit Title
# ------------------------------
st.title("Brain Tumor Detection using Deep Learning")

# ------------------------------
# Show Model Performance
# ------------------------------
st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
col4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")

# ------------------------------
# Upload MRI Image
# ------------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

# ------------------------------
# Image Preprocessing
# ------------------------------
def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((128,128))

    img_array = np.array(image) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ------------------------------
# Class Labels
# ------------------------------
class_labels = ['pituitary','glioma','notumor','meningioma']

# ------------------------------
# Prediction
# ------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    img_array = preprocess_image(image)

    prediction = model.predict(img_array)

    predicted_class = class_labels[np.argmax(prediction)]

    confidence = np.max(prediction)

    st.subheader("Prediction")

    if predicted_class == "notumor":
        st.success("No Tumor Detected")
    else:
        st.error(f"Tumor Detected: {predicted_class}")

    st.write(f"Confidence: {confidence*100:.2f}%")

    # Show probability distribution
    st.subheader("Prediction Probabilities")

    for i, label in enumerate(class_labels):
        st.progress(float(prediction[0][i]))
        st.write(f"{label}: {prediction[0][i]*100:.2f}%")
