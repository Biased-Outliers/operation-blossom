# This is the FRONT OF THE APPLICATION
# Imports
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import TFViTForImageClassification, AutoImageProcessor
from pandas import DataFrame
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

    d = DataFrame([categories, probabilities.reshape(-1,1)]).T
    d.columns = ["Flower", "Confidence"]
    d.sort_values(by='Confidence', inplace=True, ascending=False)
    d["Confidence"] = d["Confidence"].apply(lambda row: f"{row[0] * 100:.1f}%")

    return d.reset_index(drop=True)


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

    d = DataFrame([categories, pred.reshape(-1,1)]).T
    d.columns = ["Flower", "Confidence"]
    d.sort_values(by='Confidence', inplace=True, ascending=False)
    d["Confidence"] = d["Confidence"].apply(lambda row: f"{row[0] * 100:.1f}%")
    
    return d.reset_index(drop=True)

st.title('Blossom :blossom:!')

file = st.file_uploader('Upload An Image')
if file:  # if user uploaded file
    image = Image.open(file)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(image)

    with col3:
        st.write(' ')
    
    # st.image(image)
    st.header("Prediction Results :muscle:")

    column1, column2 = st.columns(2)

    with column1:
        cnn_predictions = cnn_predict(image)
        st.subheader("Convolutional Neural Network")
        result = f"{cnn_predictions['Flower'][0]}"
        st.write(f"Prediction: {cnn_predictions['Flower'][0].upper()}")
        st.table(data = cnn_predictions)
        st.write()

    with column2:    
        vit_predictions = vit_predict(image)
        st.subheader("Vision Transformer")
        st.write(f"Prediction: {vit_predictions['Flower'][0].upper()}")
        st.table(data = vit_predictions)
        st.write()

# What the hell is this? # ME
#   Provide in details what is happening.
#   Explain what the app is about
#   How To?

# Training group of flowers
# Show past pictures # LAST
# Information about the flower
#   Provide details of the flower # chatGPT
# Help Button
# Is this prediction correct?

# ADD A README SON!
#   ADD GIF of instructions
#   TO-DOs (future mods)
#   Highlight Vision Transformer
#   Why did we use streamlit?
        



    