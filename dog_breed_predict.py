import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
model_path = "dog_breed_model.keras"

if not os.path.exists(model_path):

    file_id = "1tOxHB1dVljXXO3sroJCjV-tyWDLHrT9u"

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, model_path, quiet=False)



@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

class_names = ['french_bulldog', 'german_shepherd', 'golden_retriever', 'poodle', 'yorkshire_terrier']
st.title("🐶 Dog Breed Classifier")
st.info(
    "Please upload only one of these dog breeds:\n"
    "- french_bulldog\n"
    "- german_shepherd\n"
    "- golden_retriever\n"
    "- poodle\n"
    "- yorkshire_terrier"
)
uploaded_file = st.file_uploader(
    "Upload a dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    # Resize image
    img = image.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)

    predicted_breed = class_names[predicted_index]



    st.success(f"Predicted Breed: {predicted_breed}")











