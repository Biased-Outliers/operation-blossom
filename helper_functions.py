# Imports 

import numpy as np
import matplotlib.pyplot as plt

# Visualization Functions

# Function that plots accuracy and loss of training and validation sets
def plot_history(history, epochs=10):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def plot_images(tfds, categories):
    plt.figure(figsize=(20, 10))

    for images, labels in tfds.take(1): # take(1): takes first batch from generator 
        for i in range(32):
            ax = plt.subplot(8, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(categories[np.argmax(labels[i])])
            plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_actual_prediction(model, categories, validation_set):

    plt.figure(figsize=(20, 10))
    for images, labels in validation_set.take(1):
        for i in range(15):
            ax = plt.subplot(3, 5, i + 1)
            
            img_array = images[i].numpy().astype("uint8")
            prediction = model.predict(np.array([img_array]))
            prediction_name = categories[np.argmax(prediction)]
            real_name = categories[np.argmax(labels[i])]
            
            plt.imshow(img_array)
            if prediction_name == real_name:
                plt.title(f'real: {real_name}\npred:{prediction_name}', fontdict={'color': 'g'})
            else:
                plt.title(f'real: {real_name}\npred:{prediction_name}', fontdict={'color': 'r'})
            
    plt.axis("off")