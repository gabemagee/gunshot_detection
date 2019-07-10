#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# ### File Directory Libraries

# In[ ]:


import os


# ### Math Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# ### Data Pre-Processing Libraries

# In[ ]:


import pandas as pd
import librosa
import soundfile
import re
import cv2
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer


# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# # Initialization of Variables

# In[ ]:


samples = []
labels = []
gunshot_frequency_threshold = 0.25
sample_rate = 22050
sample_rate_per_two_seconds = 44100
base_dir = "/home/amorehe/Datasets/"
data_dir = base_dir + "REU_Samples_and_Labels/"
sound_data_dir = data_dir + "Samples/"


# ## Model Parameters

# In[ ]:


number_of_epochs = 100
batch_size = 32
optimizer = optimizers.Adam(lr = 0.001, decay = 0.001 / 100)
input_tensor = Input(shape = (44100, 1))


# ## Model Architecture

# In[ ]:


x = layers.Conv1D(16, 9, activation = "relu", padding = "same")(input_tensor)
x = layers.Conv1D(16, 9, activation = "relu", padding = "same")(x)
x = layers.MaxPool1D(16)(x)
x = layers.Dropout(rate = 0.25)(x)

x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate = 0.25)(x)

x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate = 0.25)(x)

x = layers.Conv1D(256, 3, activation = "relu", padding = "same")(x)
x = layers.Conv1D(256, 3, activation = "relu", padding = "same")(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(rate = (0.5))(x) # Increasing drop-out rate here to prevent overfitting

x = layers.Dense(64, activation = "relu")(x)
x = layers.Dense(1028, activation = "relu")(x)
output_tensor = layers.Dense(2, activation = "softmax")(x)

model = tf.keras.Model(input_tensor, output_tensor)
model.compile(optimizer = optimizer, loss = keras.losses.binary_crossentropy, metrics = ["accuracy"])

model = load_model(base_dir + "gunshot_sound_model.h5")

# ## Converting model to TensorFlow Lite format


# In[ ]:

model_name = base_dir + "gunshot_sound_model"
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model_name + ".h5")
converter.post_training_quantize = True
tflite_model = converter.convert()
open(model_name + ".tflite", "wb").write(tflite_model)