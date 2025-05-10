import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model(r'D:\Nitesh Singh\MajorProject\brain_tumor_model.h5')
class_names = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

st.title("Brain Tumor Classification App")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### Predicted: {predicted_class}")