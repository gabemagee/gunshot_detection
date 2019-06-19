#!/usr/bin/env python
# coding: utf-8

# In[48]:


import os
import pandas as pd
import librosa
import librosa.display
import glob 
import pandas as pd
import numpy as np
import sys
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import optimizers
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from scipy import signal
from scipy.io import wavfile
import csv
import IPython.display as ipd
from os import listdir
from os.path import isfile, join
from glob import glob
import IPython


# In[49]:


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
    
def add_background(wav, file, data_directory, label_to_avoid):
    label_csv = data_directory + "train.csv"
    sound_directory = data_directory + "Train/"
    sound_types = pd.read_csv(label_csv)
    bg_files = os.listdir(sound_directory)
    bg_files.remove(file)
    chosen_bg_file = bg_files[np.random.randint(len(bg_files))]
    jndex = int(chosen_bg_file.split('.')[0])
    while sound_types.loc[sound_types["ID"] == jndex, "Class"].values[0] == label_to_avoid:
        chosen_bg_file = bg_files[np.random.randint(len(bg_files))]
        jndex = int(chosen_bg_file.split('.')[0])
    bg, sr = librosa.load(sound_directory + chosen_bg_file)
    ceil = max((bg.shape[0] - wav.shape[0]), 1)
    start_ = np.random.randint(ceil)
    bg_slice = bg[start_ : start_ + wav.shape[0]]
    if bg_slice.shape[0] < wav.shape[0]:
        pad_len = wav.shape[0] - bg_slice.shape[0]
        bg_slice = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len / 2)), bg_slice, np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2)))]
    wav_with_bg = wav * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.5)
    return wav_with_bg


# In[60]:


def make_spectrogram(y,sr):
    return np.array(librosa.feature.melspectrogram(y=a, sr=sr))


# In[61]:


data_directory = "/home/gamagee/workspace/gunshot_detection/REU_Data/REU_Samples_and_Labels/"
label_csv = data_directory + "labels.csv"
sample_directory = data_directory + "Samples/"


# In[ ]:


s = []
d = {}
with open(label_csv,"r") as lblcsv:
    c = list(csv.reader(lblcsv))
    header = c[0]
    for row in c[1:]:
        e["label"] = row[1]
        e["source"] = row[2]
        d[row[0]] = e
print(d)


# In[ ]:




