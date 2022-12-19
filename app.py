# Imports
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import TFViTForImageClassification, AutoImageProcessor
# import datasets
import streamlit as st

categories = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']

@st.cache()
def vit_predict(image):

    # Loading model
    vit_model = TFViTForImageClassification.from_pretrained('taraqur/blossom-vit')

    # ViT Preprocessing
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    inputs = image_processor(image, return_tensors="tf")
    
    # Prediction
    logits = vit_model(**inputs).logits

    probabilities = np.exp(logits)/np.sum(np.exp(logits))

    class_prediction = np.argmax(probabilities)

    return (categories[class_prediction], np.max(probabilities))


@st.cache()
def cnn_predict(image):

    # Loading model
    cnn_model = tf.keras.models.load_model("./saved_model/transfer_model.h5")    

    # CNN Preprocessing
    image = image.resize((224,224))
    arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image), axis=0)
    
    # Prediction
    pred = cnn_model.predict(arr)

    class_prediction = np.argmax(pred)

    return (categories[class_prediction], np.max(pred))

st.title('Blossom!')

file = st.file_uploader('Upload An Image')

if file:  # if user uploaded file
    image = Image.open(file)
    st.image(image)
    cnn_predictions = cnn_predict(image)
    vit_predictions = vit_predict(image)
    st.write(f"The CNN predicted it to be {cnn_predictions[0]} with accuracy of {cnn_predictions[1] * 100:.1f}%")
    st.write(f"The ViT predicted it to be {vit_predictions[0]} with accuracy of {vit_predictions[1] * 100:.1f}%")


    