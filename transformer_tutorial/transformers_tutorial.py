from transformers import ViTFeatureExtractor
import tensorflow as tf

model_id = "google/vit-base-patch16-224-in21k"

feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
  # tf.keras.layers.RandomCrop(96, 96),
  tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
])

# use keras image data augementation processing
def augmentation(examples):
    # print(examples["img"])
    examples["pixel_values"] = [data_augmentation(image) for image in examples["img"]]
    return examples


# basic processing (only resizing)
def process(examples):
    examples.update(feature_extractor(examples['img'], ))
    return examples

processed_dataset = eurosat_ds.map(process, batched=True)
processed_dataset