import onnxruntime as rt
import tensorflow as tf
import numpy as np

import tf2onnx

# model = tf.keras.models.load_model("./saved_model/transfer_model.h5")

# spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
# output_path = "temp" + ".onnx"

# model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
# output_names = [n.name for n in model_proto.graph.output]

# print(output_names)
providers = ['CPUExecutionProvider']
m = rt.InferenceSession("./saved_model/model.onnx", providers=providers)
output_names = [n.name for n in m.get_outputs()]
onnx_pred = m.run(output_names, {"input": x})