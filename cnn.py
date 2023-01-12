# %% [markdown]
# # Imports

# %%
# System Imports
import os
import time

# Data Imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy as sp
import cv2

import helper_functions as hf

# Deep Learning Framework
import tensorflow as tf

# SSL
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# %%
for device in tf.config.list_physical_devices('GPU'):
    print(f"* {device}")

# %% [markdown]
# ## Reading in Images

# %%
# Setting paths

train_data_path = './data/archive/train'
test_data_path = './data/archive/test'

# %%
# Viewing the classes

categories = os.listdir(train_data_path)
print(f"The classes are: {categories}")

# %%
# Configurations

SEED = 0
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
EPOCHS = 10

# %%
# Creating the training set

training_set = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_path,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=categories
)

# %%
# Creating the validation set

validation_set = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_path,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    class_names=categories
)

# %% [markdown]
# ## Plotting Images

# %%
# Plotting some images

# hf.plot_images(training_set, categories)

# %% [markdown]
# # Modeling

# %%
# Function that creates a model

def get_baseline_model(): 

    ## Clearing backend
    tf.keras.backend.clear_session()

    ## Input Layer
    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Rescaling the images
    x = tf.keras.layers.Rescaling(1./255)(inputs)

    ## First CNN
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        strides=(1,1),
        activation='relu',
    )(x)

    x = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2)
    )(x)

    ## Second CNN
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        strides=(1,1),
        activation='relu',
    )(x)

    x = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2)
    )(x)

    ## Third CNN
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3,3),
        strides=(1,1),
        activation='relu',
    )(x)

    x = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2)
    )(x)

    ## Flatten layer
    x = tf.keras.layers.Flatten()(x)

    ## First Dense layer
    x = tf.keras.layers.Dense(
        units=64,
        activation='relu'
    )(x)

    ## Output
    outputs = tf.keras.layers.Dense(
        units=len(categories),
        activation='softmax'
    )(x)

    ## Creating Model
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    ## Compiling the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    ## Viewing the architecture
    model.summary()

    return model

# %%
model_elapsed_time = {}

# %%
# # Training the model

# start_time = time.time()

# baseline_model = get_baseline_model()

# history = baseline_model.fit(
#   training_set,
#   validation_data=validation_set,
#   epochs=EPOCHS
# )

# elasped_time = time.time() - start_time
# model_elapsed_time["baseline"] = elasped_time

# %%
# # Viewing the results of the training

# hf.plot_history(history)

# %%
# hf.plot_actual_prediction(baseline_model, categories, validation_set)

# %% [markdown]
# Model is not performing well. Accuracy is terrible (bias is high). And generalization is bad (variance is high). 
# 
# Since the model did not accurately predict the flowers, the model didn't learn key features that differentiates the flowers (categories).
# 
# Solutions:
# <ul>
#     <li> Data Augmentation </li>
#          * Create new images by augmenting the images to expose the model to more images
#     <li> Transfer Learning</li>
#         * Use a working model that performs well for our task<br>
#     <li> Get more data</li>
#         * Find more data for the model
# </ul>

# %% [markdown]
# ## Data Augmentation

# %%
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip(
      "horizontal_and_vertical", 
      input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[0], 3)
  ),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
  tf.keras.layers.RandomCrop(
    128, 128
  ),
  tf.keras.layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[0])
])

def prepare(ds, shuffle=False, augment=False):

    if shuffle:
        ds = ds.shuffle(1000)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y), 
            num_parallel_calls=AUTOTUNE
        )

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

# %%
train_ds = prepare(training_set, shuffle=True, augment=True)
val_ds = prepare(validation_set)

# %%
# # Running the model

# tf.keras.backend.clear_session()

# start_time = time.time()

# augment_model = get_baseline_model()

# augment_history = augment_model.fit(
#     train_ds,
#     validation_data = val_ds,
#     epochs=EPOCHS
# )

# elasped_time = time.time() - start_time
# model_elapsed_time['data_augmentation'] = elasped_time

# %%
# # Viewing the results of the training

# hf.plot_history(augment_history)

# %%
# augment_model.save('saved_model/augment_model.h5')

# %% [markdown]
# The training and validation have similar accuracy and loss (low variance). However the accuracy is ~70% (high bias). If we would like, we can do more augmentation like random contrasting.
# 
# Next we will try transfer learning. Use a model that has a great track record classifying and apply it to our task.

# %% [markdown]
# ## Transfer Learning

# %%
# base_transfer_model = tf.keras.applications.MobileNetV3Small(
#     input_shape=IMAGE_SIZE+ (3,),
#     include_top=False,
#     weights='imagenet',
# )

# base_transfer_model.trainable = False

# base_transfer_model.summary()

# %%
# image_batch, label_batch = next(iter(train_ds))
# feature_batch = base_transfer_model(image_batch)
# print(feature_batch.shape)

# %%
# tf.keras.backend.clear_session()
# inputs = tf.keras.Input(shape=(224, 224, 3))
# x = data_augmentation(inputs)
# x = base_transfer_model(x, training=False)
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# outputs = tf.keras.layers.Dense(len(categories), activation='softmax')(x)
# transfer_model = tf.keras.Model(inputs, outputs)

# %%
# # start_time = time.time()

# transfer_model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.CategoricalCrossentropy(),
#     metrics=['accuracy']
# )

# # elasped_time = time.time() - start_time
# # model_elapsed_time['transfer_model'] = elasped_time

# %%
# transfer_model.summary()

# %%
# transfer_history = transfer_model.fit(
#     train_ds,
#     validation_data = val_ds,
#     epochs=EPOCHS
# )

# %%
# # Let's take a look to see how many layers are in the base model
# print("Number of layers in the base model: ", len(base_transfer_model.layers))

# # Fine-tune from this layer onwards
# fine_tune_at = 200

# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_transfer_model.layers[fine_tune_at:]:
#     layer.trainable = True

# %%
# transfer_model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.CategoricalCrossentropy(),
#     metrics=['accuracy']
# )

# transfer_fine = transfer_model.fit(
#     train_ds,
#     epochs=EPOCHS+10,
#     initial_epoch=transfer_history.epoch[-1],
#     validation_data=val_ds
# )

# %%
# hf.plot_history(transfer_fine)

# %%
# transfer_model.save('saved_model/transfer_model.h5')

# %%
# from tenkeras.models import load_model
loaded_model = tf.keras.models.load_model("./saved_model/transfer_model.h5")

# %%
from PIL import Image

new_image_path = './data/FREEDOM.jpg'
image = Image.open(new_image_path).resize((224,224))

# %%
arr = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image), axis=0)

# %%
pred = loaded_model.predict(arr)

# %%
np.argmax(pred, axis=1)

# %%
resnet_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=IMAGE_SIZE+ (3,),
    include_top=False,
    weights='imagenet',
)

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(resnet_model.layers))



# resnet_model.summary()

# %%
resnet_model.trainable = False

# Fine-tune from this layer onwards
fine_tune_at = 170

# Freeze all the layers before the `fine_tune_at` layer
for layer in resnet_model.layers[fine_tune_at:]:
    layer.trainable = True

# %%
tf.keras.backend.clear_session()
inputs = tf.keras.Input(shape=(224, 224, 3))

# x = data_augmentation(inputs)
# x = resnet_model(x, training=False)

x = resnet_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(len(categories), activation='softmax')(x)
transfer_resnet_model = tf.keras.Model(inputs, outputs)

# %%
transfer_resnet_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,
    patience=5, 
    min_lr=0.001
)

es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=3,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
)

transfer_fine = transfer_resnet_model.fit(
    train_ds,
    epochs=20,
    # initial_epoch=transfer_history.epoch[-1],
    validation_data=val_ds,
    callbacks=[reduce_lr, es]
    
)

# %%



