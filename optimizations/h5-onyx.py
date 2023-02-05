# from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model
# import onnx
# import keras2onnx
import tensorflow as tf

onnx_model_name = './saved_model/vit_model.onnx'

model = tf.keras.models.load_model('saved_model/local-tf-checkpoint/tf_model.h5')
# onnx_model = keras2onnx.convert_keras(model, model.name)
# onnx.save_model(onnx_model, onnx_model_name)
model.save("./saved_model/vit")

# python3 -m tf2onnx.convert --saved-model saved_model/transfer_model --output saved_model/model.onnx