import tensorflow as tf
from transformers import AutoFeatureExtractor
from pandas import DataFrame
import numpy as np
import onnxruntime as rt
import onnx
from transformers.onnx.features import FeaturesManager

categories = ['daisy', 'rose', 'tulip', 'dandelion', 'sunflower']

def vit_predict(image):

    # ViT Preprocessing
    image_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    inputs = image_processor(image, return_tensors="np")
    
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession("./saved_model/tf-vit-model.onnx", providers=providers)
    input_name = m.get_inputs()[0].name
    output_names = [n.name for n in m.get_outputs()]
    pred = m.run(output_names, {input_name: inputs['pixel_values']})
    class_prediction = np.argmax(pred[0])

    probabilities = np.exp(pred[0])/np.sum(np.exp(pred[0]))

    d = DataFrame([categories, probabilities.reshape(-1,1)]).T
    d.columns = ["Flower", "Confidence"]
    d.sort_values(by='Confidence', inplace=True, ascending=False)
    d["Confidence"] = d["Confidence"].apply(lambda row: f"{row[0] * 100:.1f}%")

    return d.reset_index(drop=True)

def cnn_predict(image):

    # CNN Preprocessing
    image = image.resize((224,224))
    arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image), axis=0)

    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession("./saved_model/model.onnx", providers=providers)
    input_name = m.get_inputs()[0].name
    output_names = [n.name for n in m.get_outputs()]
    pred = m.run(output_names, {input_name: arr})
    class_prediction = np.argmax(pred[0])

    d = DataFrame([categories, pred[0].reshape(-1,1)]).T
    d.columns = ["Flower", "Confidence"]
    d.sort_values(by='Confidence', inplace=True, ascending=False)
    d["Confidence"] = d["Confidence"].apply(lambda row: f"{row[0] * 100:.1f}%")
    
    return d.reset_index(drop=True)