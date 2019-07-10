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


# ### Configuration of Imported Libraries

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')


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


# ## Loading augmented sample file and label file as numpy arrays

# In[ ]:

samples = np.load(base_dir + "gunshot_augmented_sound_samples.npy")
labels = np.load(base_dir + "gunshot_augmented_sound_labels.npy")


# ## Restructuring the label data

# In[ ]:


labels = np.array([("gun_shot" if label == 1 else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels, 1 - labels))


# ### Optional debugging of the label data's shape

# In[ ]:


print(labels.shape)


# ## Arranging the data

# In[ ]:


kf = KFold(n_splits = 3, shuffle = True)
samples = np.array(samples)
labels = np.array(labels)
for train_index, test_index in kf.split(samples):
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]


# ## Reshaping the sound data

# In[ ]:


train_wav = train_wav.reshape(-1, sample_rate_per_two_seconds, 1)
test_wav = test_wav.reshape(-1, sample_rate_per_two_seconds, 1)


# ### Optional debugging of the sound data's shape

# In[ ]:


print(train_wav.shape)


# # Model


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
model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])


# ## Configuring model properties

# In[ ]:


model_filename = base_dir + "gunshot_sound_model.pkl"

model_callbacks = [
    EarlyStopping(monitor = 'val_acc',
                  patience = 15,
                  verbose = 1,
                  mode = 'max'),

    ModelCheckpoint(model_filename, monitor = 'val_acc',
                    verbose = 1,
                    save_best_only = True,
                    mode = 'max')
]


# ### Optional debugging of the model's architecture

# In[ ]:


model.summary()


# ## Training & caching the model

# In[ ]:


History = model.fit(train_wav, train_label,
          validation_data = [test_wav, test_label],
          epochs = number_of_epochs,
          callbacks = model_callbacks,
          verbose = 1,
          batch_size = batch_size,
          shuffle = True)

model.save(base_dir + "gunshot_sound_model.h5")


# ### Optional debugging of incorrectly-labeled examples

# In[ ]:


y_test_pred = model.predict(test_wav)
y_predicted_classes_test = y_test_pred.argmax(axis = -1)
y_actual_classes_test = test_label.argmax(axis = -1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)


# ## Converting model to TensorFlow Lite format


# In[ ]:

model_name = base_dir + "gunshot_sound_model"
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model_name + ".h5")
converter.post_training_quantize = True
tflite_model = converter.convert()
open(model_name + ".tflite", "wb").write(tflite_model)