#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import os

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import librosa
import soundfile
import re
import cv2
from sklearn.model_selection import KFold

import IPython.display as ipd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# In[4]:


samples=[]
labels = []
gunshot_frequency_threshold = 0.25
sample_rate = 22050
sample_rate_per_two_seconds = 44100
input_shape = (sample_rate_per_two_seconds, 1)
base_dir = "/home/lauogden/data/"
data_dir = base_dir + "REU_Samples_and_Labels/"
sound_data_dir = data_dir + "Samples/"


# In[5]:


#load samples and labels
samples = np.load(base_dir + "gunshot_sound_samples_multiclass.npy")
labels = np.load(base_dir + "gunshot_sound_labels_multiclass.npy")


# In[6]:


#to categorical
#reminder, this is for MULTICLASS CLASSIFICATION:
    #0: miscellanious
    #1: gunshot
    #2: fireworks
    #3: glassbreak

labels = keras.utils.to_categorical(labels, 4)


# In[7]:


print(labels.shape)


# In[8]:

#SWITCHING TO THE GPU
print("...switching to the gpu...")
with tf.device("/gpu:0"):
    #train test split
    kf = KFold(n_splits=3, shuffle=True)
    samples = np.array(samples)
    labels = np.array(labels)
    for train_index, test_index in kf.split(samples):
        train_wav, test_wav = samples[train_index], samples[test_index]
        train_label, test_label = labels[train_index], labels[test_index]


    # In[9]:


    #reshape the data
    train_wav = train_wav.reshape(-1, sample_rate_per_two_seconds, 1)
    test_wav = test_wav.reshape(-1, sample_rate_per_two_seconds, 1)


    # In[10]:


    print(train_wav.shape)


    # In[11]:


    #AUC metric
    def auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc


    # In[17]:


    #Model parameters
    drop_out_rate = 0.1
    learning_rate = 0.001
    number_of_epochs = 100
    number_of_classes = 4
    batch_size = 32
    optimizer = optimizers.Adam(learning_rate, learning_rate / 100)
    input_tensor = Input(shape=input_shape)
    metrics = [auc, "accuracy"]


    # In[18]:


    #architecture
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
    model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=metrics)


    # In[19]:


    #model properties

    model_filename = base_dir + "gunshot_sound_model_multiclass.pkl"

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


    # In[20]:


    model.summary()


    # In[21]:


    #TRAIN IT
    History = model.fit(train_wav, train_label, 
              validation_data=[test_wav, test_label],
              epochs=number_of_epochs,
              callbacks=model_callbacks,
              verbose=1,
              batch_size=batch_size,
              shuffle=True)

    model.save(base_dir + "gunshot_sound_model_multiclass.h5")






