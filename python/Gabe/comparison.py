import os
import glob
import numpy as np
import sys
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling1D, MaxPooling1D, Dense, Dropout, Activation, Flatten
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
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import device_lib
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
# In[8]:

print("available gpus:",get_available_gpus())


base_dir = "/home/gamagee/workspace/gunshot_detection/"
model_dir = base_dir+"raspberry_pi/models/"

sample_dir = base_dir+"REU_Data/spectrogram_training/samples_and_labels/"

label_path = sample_dir+"gunshot_augmented_sound_labels.npy"

#sample_path = sample_dir+"gunshot_augmented_sound_samples.npy"

sample_path = sample_dir+"gunshot_augmented_sound_samples_spectro.npy"

spectrograph_samples_2_fn = sample_dir+"spectrogram_samples_power_to_db.npy"

samples = np.load(sample_path)
labels = np.load(label_path)

sample_weights = np.array([1 for normally_recorded_sample in range(len(samples) - 660)] + [SELF_RECORDING_WEIGHT for raspberry_pi_recorded_sample in range(660)])


print(samples.shape)

samples.reshape(-1,128,87,1)
sample_rate_per_two_seconds = 44100
number_of_classes = 2
sr = 22050
input_shape = (128, 87, 1)

print(labels.shape)


testing_indexes_path = base_dir+"raspberry_pi/indexes/testing_set_indexes.npy"

testing_indexes = np.load(testing_indexes_path)

training_indexes_path = base_dir+"raspberry_pi/indexes/training_set_indexes.npy"

training_indexes = np.load(training_indexes_path)

labels = keras.utils.to_categorical(labels, 2)

print(labels.shape)

#sample_weights = np.array( [1 for normally_recorded_sample in range(len(samples) - 660)] + [50 for raspberry_pi_recorded_sample in range(660)])
print("Shape of samples weights before splitting:", sample_weights.shape)

print("~~~~~~~~~~~~~~~~")

train_wav = []
train_label = []
train_weights = []
test_wav = []
test_label = []
test_weights = []
validation_wav = []
validation_label = []
validation_weights = []

for i in range(len(labels)):
    if i in training_indexes:
        train_wav.append(samples[i])
        train_label.append(labels[i])
        train_weights.append(sample_weights[i])
    elif i in testing_indexes:
        test_wav.append(samples[i])
        test_label.append(labels[i])
        test_weights.append(sample_weights[i])
    else:
        validation_wav.append(samples[i])
        validation_label.append(labels[i])
        validation_weights.append(sample_weights[i])

train_wav = np.array(train_wav)
train_label = np.array(train_label)
train_weights = np.array(train_weights)
test_wav = np.array(test_wav)
test_label = np.array(test_label)
test_weights = np.array(test_weights)
validation_wav = np.array(validation_wav)
validation_label = np.array(validation_label)
validation_weights = np.array(validation_weights)

print("finished split")



print("loaded models")


for i in range(len(validation_wav)):
    print(i)
    x = validation_wav[i]
    y = validation_label[i]
