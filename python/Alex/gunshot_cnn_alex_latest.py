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
import librosa
import re
import cv2
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
from tensorflow.keras import Input, layers, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# # Initialization of Variables

# In[ ]:


samples=[]
labels = []
gunshot_frequency_threshold = 0.25
sample_rate = 22050
sample_rate_per_two_seconds = 44100
input_shape = (sample_rate_per_two_seconds, 1)


# # Data Pre-Processing


# ## Loading sample file and label file as numpy arrays

# In[ ]:


samples = np.load("/home/alexm/Datasets/gunshot_sound_samples.npy")
labels = np.load("/home/alexm/Datasets/gunshot_sound_labels.npy")


# ## Data augmentation functions

# In[ ]:


def time_shift(wav):
    start_ = int(np.random.uniform(-wav.shape[0] * 0.5, wav.shape[0] * 0.5))
    if start_ >= 0:
        wav_time_shift = np.r_[wav[start_:], np.random.uniform(-0.001, 0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001, 0.001, -start_), wav[:start_]]
    return wav_time_shift
    
def change_pitch(wav, sample_rate):
    magnitude = int(np.random.uniform(-10, 10))
    wav_pitch_change = librosa.effects.pitch_shift(wav, sample_rate, magnitude)
    return wav_pitch_change
    
def speed_change(wav):
    speed_rate = np.random.uniform(0.7, 1.3)
    wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
    
    if len(wav_speed_tune) < len(wav):
        pad_len = len(wav) - len(wav_speed_tune)
        wav_speed_tune = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len / 2)),
                               wav_speed_tune,
                               np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2)))]
    else: 
        cut_len = len(wav_speed_tune) - len(wav)
        wav_speed_tune = wav_speed_tune[int(cut_len / 2) : int(cut_len / 2) + len(wav)]
    return wav_speed_tune
    
def change_volume(wav, magnitude):
    # 0 < x < 1 quieter; x = 1 identity; x > 1 louder
    wav_volume_change = np.multiply(np.array([magnitude]), wav)
    return wav_volume_change
    
def add_background(wav, sound_directory):
    bg_files = os.listdir(sound_directory)
    bg_files.remove(chosen_file)
    chosen_bg_file = bg_files[np.random.randint(len(bg_files))]
    bg, sr = librosa.load(sound_directory + chosen_bg_file, sr=None)
    ceil = max((bg.shape[0] - wav.shape[0]), 1)
    start_ = np.random.randint(ceil)
    bg_slice = bg[start_ : start_+ wav.shape[0]]
    if bg_slice.shape[0] < wav.shape[0]:
        pad_len = wav.shape[0] - bg_slice.shape[0]
        bg_slice = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len / 2)), bg_slice, np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2)))]
    wav_with_bg = wav * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.5)
    return wav_with_bg


# ## Augmenting data (time shifting and speed changing)

# In[ ]:


augmented_samples = np.zeros((samples.shape[0] * 6, samples.shape[1]))
augmented_labels = np.zeros((labels.shape[0] * 6,))
j = 0

for i in range (0, len(augmented_samples), 6):
    augmented_samples[i,:] = samples[j,:]
    augmented_samples[i + 1,:] = time_shift(samples[j,:])
    augmented_samples[i + 2,:] = change_pitch(samples[j,:], sample_rate)
    augmented_samples[i + 3,:] = speed_change(samples[j,:])
    augmented_samples[i + 4,:] = change_volume(samples[j,:], np.random.uniform())
    augmented_samples[i + 5,:] = add_background(samples[j,:], sound_data_dir)
    
    augmented_labels[i] = labels[j]
    augmented_labels[i + 1] = labels[j]
    augmented_labels[i + 2] = labels[j]
    augmented_labels[i + 3] = labels[j]
    augmented_labels[i + 4] = labels[j]
    augmented_labels[i + 5] = labels[j]
    j += 1

samples = augmented_samples
labels = augmented_labels

print("The number of samples of available for training is currently " + str(len(samples)) + '.')
print("The number of labels of available for training is currently " + str(len(labels)) + '.')


# ## Saving augmented samples and labels as numpy array files

# In[ ]:


np.save("/home/alexm/Datasets/gunshot_augmented_sound_samples.npy", samples)
np.save("/home/alexm/Datasets/gunshot_augmented_sound_labels.npy", labels)


# ## Restructuring the label data

# In[ ]:


labels = keras.utils.to_categorical(labels, 2)


# ### Optional debugging of the label data's shape

# In[ ]:


print(labels.shape)


# ## Arranging the data

# In[ ]:


kf = KFold(n_splits=3, shuffle=True)
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


# ## Model Parameters

# In[ ]:


drop_out_rate = 0.1
learning_rate = 0.001
number_of_epochs = 100
batch_size = 32


# ## ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"

# In[ ]:


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# ## Model Architecture

# In[ ]:


input_tensor = Input(shape=input_shape)
number_of_classes = 2

x = layers.Conv1D(16, 9, activation="relu", padding="same")(input_tensor)
x = layers.Conv1D(16, 9, activation="relu", padding="same")(x)
x = layers.MaxPool1D(16)(x)
x = layers.Dropout(rate=drop_out_rate)(x)

x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate=drop_out_rate)(x)

x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate=drop_out_rate)(x)

x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(rate=(drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting

x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(1028, activation="relu")(x)
output_tensor = layers.Dense(number_of_classes, activation="softmax")(x)

model = tf.keras.Model(input_tensor, output_tensor)

optimizer = optimizers.Adam(learning_rate, learning_rate / 100)

model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=[auc])


# ## Configuring model properties

# In[ ]:


model_filename = '/home/alexm/Datasets/gunshot_sound_model.pkl'

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
          epochs=number_of_epochs,
          callbacks=model_callbacks,
          verbose=1,
          batch_size=batch_size,
          shuffle=True)

model.save("/home/alexm/Datasets/gunshot_sound_model.h5")


# ### Optional debugging of incorrectly-labeled examples

# In[ ]:


y_test_pred = model.predict(test_wav)
y_predicted_classes_test = y_test_pred.argmax(axis=-1)
y_actual_classes_test= test_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)
