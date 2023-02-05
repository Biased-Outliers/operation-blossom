from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoFeatureExtractor
from pathlib import Path


model_id="taraqur/blossom-vit"
onnx_path = Path("optimum-onnx")

# load vanilla transformers and convert to onnx
model = ORTModelForImageClassification.from_pretrained(model_id, from_transformers=True)
preprocessor = AutoFeatureExtractor.from_pretrained(model_id)

# save onnx checkpoint and tokenizer
model.save_pretrained(onnx_path)
preprocessor.save_pretrained(onnx_path)

from transformers import pipeline

vanilla_clf = pipeline("image-classification", model=model, feature_extractor=preprocessor)
