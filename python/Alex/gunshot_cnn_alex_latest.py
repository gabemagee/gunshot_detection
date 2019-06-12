#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# ### File Directory Libraries

# In[ ]:


import glob
import os
from os.path import isdir, join
from pathlib import Path


# ### Math Libraries

# In[ ]:


import numpy as np
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


# ### Data Pre-Processing Libraries

# In[ ]:


import pandas as pd
from sklearn.model_selection import KFold


# ### Visualization Libraries

# In[ ]:


import seaborn as sns
import IPython.display as ipd
import librosa.display


# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ### Configuration of Imported Libraries

# In[ ]:


py.init_notebook_mode(connected=True)


# # Initialization of Variables

# In[ ]:


samples=[]
labels = []
sampling_rate_per_two_seconds = 44100
input_shape = (sampling_rate_per_two_seconds, 1)


# ## Loading sample file and label file as numpy arrays

# In[ ]:


samples = np.load("/home/amorehe/Datasets/gunshot_sound_samples.npy")
labels = np.load("/home/amorehe/Datasets/gunshot_sound_labels.npy")


# ## Arranging the data

# In[ ]:


kf = KFold(n_splits=3, shuffle=True)
samples = np.array(samples)
labels = np.array(labels)
for train_index, test_index in kf.split(samples):
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]


# ## Reshaping/restructuring the data

# In[ ]:


train_wav = train_wav.reshape(-1, sampling_rate_per_two_seconds, 1)
test_wav = test_wav.reshape(-1, sampling_rate_per_two_seconds, 1)
train_label = keras.utils.to_categorical(train_label, 2)
test_label = keras.utils.to_categorical(test_label, 2)


# ### Optional debugging of the training data's shape

# In[ ]:


print(train_wav.shape)


# # Model

# ## Loading previous model

# In[ ]:


model = load_model("/home/amorehe/Datasets/gunshot_sound_full_model.h5")


# ## Model Parameters

# In[ ]:


learning_rate = 0.001
batch_size = 32
drop_out_rate = 0.2


# ## Model Architecture

# In[ ]:


input_tensor = Input(shape=input_shape)

x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_tensor)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(4)(x)
x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(4)(x)
x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(6)(x)
x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(6)(x)
x = layers.Conv1D(256, 3, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(6)(x)
x = layers.Flatten()(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(50, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(20, activation='relu')(x)

output_tensor = layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(input_tensor, output_tensor)

model.compile(loss=keras.losses.binary_crossentropy,
             optimizer=keras.optimizers.Adam(lr = learning_rate),
             metrics=['accuracy'])


# ## Configuring model properties

# In[ ]:


model_filename = '/home/amorehe/Datasets/gunshot_sound_model.pkl'

model_callbacks = [
    EarlyStopping(monitor='val_acc',
                  patience=10,
                  verbose=1,
                  mode='auto'),
    
    ModelCheckpoint(model_filename, monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    mode='auto'),
]


# ### Optional debugging of the model's architecture

# In[ ]:


model.summary()


# ## Training & caching the model

# In[ ]:


History = model.fit(train_wav, train_label, 
          validation_data=[test_wav, test_label],
          epochs=50,
          callbacks=model_callbacks,
          verbose=1,
          batch_size=batch_size,
          shuffle=True)

model.save("/home/amorehe/Datasets/gunshot_sound_full_model.h5")


# ### Optional debugging of incorrectly-labeled examples

# In[ ]:


y_test_pred = model.predict(test_wav)
y_predicted_classes_test = y_test_pred.argmax(axis=-1)
y_actual_classes_test= test_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)
