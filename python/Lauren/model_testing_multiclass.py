#!/usr/bin/env python
# coding: utf-8

# In[3]:


#imports
import glob
import os
from os.path import isdir, join
from pathlib import Path

import numpy as np
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import pandas as pd
import librosa
import re
from sklearn.model_selection import KFold

import seaborn as sns
import IPython.display as ipd
import librosa.display

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import csv


# In[4]:


#auc custom metric
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

#other things
number_of_desired_samples = 250
sampling_rate_per_two_seconds = 44100


# In[5]:


#paths

model_path = "/home/lauogden/models/gunshot_sound_model_multiclass.h5"

a_labels = "/home/lauogden/data/gunshot_augmented_sound_labels_multiclass.npy"
a_samples = "//home/lauogden/data/gunshot_augmented_sound_samples_multiclass.npy"

b_labels = "/home/lauogden/data/gunshot_augmented_sound_labels_multiclass.npy"
b_samples = "/home/lauogden/data/gunshot_augmented_sound_samples_multiclass.npy"

results = "/home/lauogden/models/testing_results/"

results_guns = results +"guns/"

results_others = results+"others/"


# In[ ]:


#text file to log wrong results
file = open(results + "incorrect_samples.txt","w")


# In[6]:


#load model
model = keras.models.load_model(model_path, custom_objects={'auc' : auc})

model.summary()

sr = 22050


# In[7]:


#load samples and labels
label_np = np.load(a_labels)
scont = np.load(a_samples)


# In[ ]:


#to categorical, reshape
#label_np_1 = np.array(keras.utils.to_categorical(label_np, 4))
sample_np = np.array(scont).reshape(-1, sampling_rate_per_two_seconds, 1)


# In[ ]:


#predict
predictions = np.argmax(model.predict(sample_np),axis=1)


# In[ ]:


#find differences
    #if diff = 0, that means the model predicted it correctly
diff = label_np - predictions


# In[ ]:


#get the indices of the samples it got wrong
indexes = []
for i in range(len(diff)):
    if diff[i]!=0:
        indexes.append(i)


# In[ ]:


#for each index, write the wav file and put it in the text file
for ind in indexes:
    if label_np[ind]==1:
        direc = results_guns
    else:
        direc = results_others
    filepath = direc+"/"+str(ind)+".wav"
    print(filepath)
    librosa.output.write_wav(filepath,scont[ind],sr)
    file.write("filename: " + str(ind) + ".wav\n")
    file.write("    actual label: " + str(label_np[ind])+ "\n")
    file.write("    predicted label: " + str(predictions[ind])+ "\n")

