#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from scipy.fftpack import fft
from scipy import signal
import librosa
import os

# Dimension Reduction
from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

# Data Pre-processing
import pandas as pd
from sklearn.model_selection import KFold
import soundfile

# Deep Learning
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

from tensorflow.python.client import device_lib
import re


# In[18]:


#read the csv file of all data labels

sound_types = pd.read_csv("/home/lauogden/data/REU_Data_organized/train.csv")


# In[19]:


urban_sound_dir = "/home/lauogden/data/REU_Data_organized/Train/"
urban_sound_iterator = 0
sampling_rate_per_two_seconds = 44100
samples = []
labels = []

all_files = os.listdir(urban_sound_dir)
all_files.sort()

for file in all_files:
    if file.endswith(".wav"):
        try:
            # Adding 2 second-long samples to the list of samples
            urban_sound_iterator = int(re.search(r'\d+', file).group())
            sample, sample_rate = librosa.load(urban_sound_dir + file)
            prescribed_label = sound_types.loc[sound_types["ID"] == urban_sound_iterator, "Class"].values[0]

            if len(sample) <= sampling_rate_per_two_seconds:
                label = 1
                number_of_missing_hertz = sampling_rate_per_two_seconds - len(sample)
                padded_sample = np.array(sample.tolist() + [0 for i in range(number_of_missing_hertz)])
                if prescribed_label != "gun_shot":
                    label = 0
                elif np.max(abs(sample)) < 0.25:
                    label = 0

                samples.append(padded_sample)
                labels.append(label)
            else:
                for i in range(0, sample.size - sampling_rate_per_two_seconds, sampling_rate_per_two_seconds):
                    sample_slice = sample[i : i + sampling_rate_per_two_seconds]
                    if prescribed_label != "gun_shot":
                        label = 0
                    elif np.max(abs(sample_slice)) < 0.25:
                        label = 0

                    samples.append(sample_slice)
                    labels.append(label)
        except:
            print("sound not recognized by Librosa:" + file)
            pass

       

print("The number of samples of available for training is currently " + str(len(samples)) + '.')
print("The number of labels of available for training is currently " + str(len(labels)) + '.')


# In[ ]:


#save it
np.save("/home/lauogden/data/sound_samples3.npy", samples)
np.save("/home/lauogden/data/sound_labels3.npy", labels)

