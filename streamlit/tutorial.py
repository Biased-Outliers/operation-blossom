# Deploying a CNN model on Streamlit
# https://analyticsindiamag.com/deploy-your-deep-learning-based-image-classification-model-with-streamlit/

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np