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
import librosa.display
import soundfile
import re
import cv2
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer


# ### Visualization Libraries

# In[ ]:


import IPython.display as ipd


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


# # Data Pre-Processing

# ## Reading in the CSV file of descriptors for many kinds of sounds

# In[ ]:


sound_types = pd.read_csv(data_dir + "labels.csv")


# ## Loading augmented sample file and label file as numpy arrays

# In[ ]:


samples = np.load(base_dir + "gunshot_augmented_sound_samples.npy")
labels = np.load(base_dir + "gunshot_augmented_sound_labels.npy")


# ## Converting Augmented Samples to Spectrograms

# ### Defining Spectrogram Conversion Functions

# In[ ]:


def convert_to_spectrogram(data, sample_rate):
    return np.array(librosa.feature.melspectrogram(y = data, sr = sample_rate))

def save_spectrogram_as_png(spectrogram, index):
    plt.interactive(False)
    fig = plt.figure(figsize = [0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref = np.max))
    plt.savefig("~/Datasets/Spectrograms/" + str(index) + ".png", dpi = 400, bbox_inches = "tight", pad_inches = 0)

    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')


# ### Iteratively Converting All Augmented Samples into Spectrograms

# In[ ]:


spectogram_index = 0

for sample in samples:
    spectrogram = convert_to_spectrogram(sample, sample_rate)
    save_spectrogram_as_png(spectrogram, spectogram_index)
    print("Successfully saved augmented sample #" + str(spectogram_index) + " as a spectrogram...")
    spectogram_index += 1
