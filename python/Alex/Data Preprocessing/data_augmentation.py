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
import six
from array import array
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer


# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
from tensorflow.keras import Input, layers, backend as K
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# # Initialization of Variables

# In[ ]:


GUNSHOT_FREQUENCY_THESHOLD = 0.25
SAMPLE_RATE_PER_SECOND = 22050
SAMPLE_RATE_PER_TWO_SECONDS = 44100
HOP_LENGTH = 345 * 2
MINIMUM_FREQUENCY = 20
MAXIMUM_FREQUENCY = SAMPLE_RATE_PER_SECOND
NUMBER_OF_MELS = 128
NUMBER_OF_FFTS = NUMBER_OF_MELS * 20
BASE_DIRECTORY = "/home/amorehe/Datasets/"
DATA_CATEGORY = "testing"


# # Data Pre-Processing

# ## Loading NumPy files as NumPy arrays

# In[ ]:


samples = np.load(BASE_DIRECTORY + DATA_CATEGORY + "_samples.npy")
labels = np.load(BASE_DIRECTORY + DATA_CATEGORY +  "_labels.npy")


# ## Data augmentation functions

# In[ ]:


def time_shift(sample):
    start_ = int(np.random.uniform(-7000, 7000))
    if start_ >= 0:
        sample_time_shift = np.r_[sample[start_:], np.random.uniform(-0.001, 0.001, start_)]
    else:
        sample_time_shift = np.r_[np.random.uniform(-0.001, 0.001, -start_), sample[:start_]]
    return sample_time_shift
    
def change_pitch(sample, sample_rate):
    magnitude = (np.random.uniform(-0.1, 0.1))
    sample_pitch_change = librosa.effects.pitch_shift(sample, sample_rate, magnitude)
    return sample_pitch_change
    
def speed_change(sample):
    speed_rate = np.random.uniform(0.7, 1.3)
    sample_speed_tune = cv2.resize(sample, (1, int(len(sample) * speed_rate))).squeeze()
    
    if len(sample_speed_tune) < len(sample):
        pad_len = len(sample) - len(sample_speed_tune)
        sample_speed_tune = np.r_[np.random.uniform(-0.0001, 0.0001, int(pad_len / 2)),
                               sample_speed_tune,
                               np.random.uniform(-0.0001, 0.0001, int(np.ceil(pad_len / 2)))]
    else: 
        cut_len = len(sample_speed_tune) - len(sample)
        sample_speed_tune = sample_speed_tune[int(cut_len / 2) : int(cut_len / 2) + len(sample)]
    return sample_speed_tune
    
def change_volume(sample, magnitude):
    # 0 < x < 1 quieter; x = 1 identity; x > 1 louder
    sample_volume_change = np.multiply(np.array([magnitude]), sample)
    return sample_volume_change
    
def add_background(sample, samples, labels, label_to_avoid):
    sample_index = [np.array_equal(sample, array) for array in samples.tolist()].index(True)
    chosen_bg_sample = samples[np.random.randint(len(samples))]
    chosen_bg_sample_index, = [np.array_equals(chosen_bg_sample, array) for array in samples.tolist()].index(True)
    while chosen_bg_sample_index == sample_index or labels[sample_index] == label_to_avoid:
        chosen_bg_sample = samples[np.random.randint(len(samples))]
        chosen_bg_sample_index = [np.array_equals(chosen_bg_sample, array) for array in samples.tolist()].index(True)
    ceil = max((chosen_bg_sample.shape[0] - sample.shape[0]), 1)
    start_ = np.random.randint(ceil)
    bg_slice = chosen_bg_sample[start_ : start_ + sample.shape[0]]
    if bg_slice.shape[0] < sample.shape[0]:
        pad_len = sample.shape[0] - bg_slice.shape[0]
        bg_slice = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len / 2)), bg_slice, np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2)))]
    sample_with_bg = sample * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.5)
    return sample_with_bg


# ## Augmenting data (i.e. time shifting, speed changing, etc.)

# In[ ]:


samples = np.array(samples)
labels = np.array(labels)
number_of_augmentations = 5
augmented_samples = np.zeros((samples.shape[0] * (number_of_augmentations + 1), SAMPLE_RATE_PER_TWO_SECONDS))
augmented_labels = np.zeros((labels.shape[0] * (number_of_augmentations + 1),))
j = 0

for i in range (0, len(augmented_samples), (number_of_augmentations + 1)):
    augmented_samples[i,:] = samples[j,:]
    augmented_samples[i + 1,:] = time_shift(samples[j,:])
    augmented_samples[i + 2,:] = change_pitch(samples[j,:], SAMPLE_RATE_PER_SECOND)
    augmented_samples[i + 3,:] = speed_change(samples[j,:])
    augmented_samples[i + 4,:] = change_volume(samples[j,:], np.random.uniform())
    
    if labels[j] == "gun_shot":
        augmented_samples[i + 5,:] = add_background(samples[j,:], samples, labels, "") 
    else:
        augmented_samples[i + 5,:] = add_background(samples[j,:], samples, labels, "gun_shot")
    
    augmented_labels[i] = labels[j]
    augmented_labels[i + 1] = labels[j]
    augmented_labels[i + 2] = labels[j]
    augmented_labels[i + 3] = labels[j]
    augmented_labels[i + 4] = labels[j]
    augmented_labels[i + 5] = labels[j]
    
    print("Finished augmenting sample #" + str(j + 1))
    j += 1

samples = augmented_samples
labels = augmented_labels

print("The number of samples available for training is currently " + str(len(samples)) + '.')
print("The number of labels available for training is currently " + str(len(labels)) + '.')


# ## Saving augmented NumPy arrays as NumPy files

# In[ ]:


np.save(BASE_DIRECTORY + "augmented_" + DATA_CATEGORY + "_samples.npy", samples)
np.save(BASE_DIRECTORY + "augmented_" + DATA_CATEGORY + "_labels.npy", labels)


# ### Debugging of the sample and label data's shape (optional)

# In[ ]:


print("Shape of samples array:", samples.shape)
print("Shape of labels array:", labels.shape)