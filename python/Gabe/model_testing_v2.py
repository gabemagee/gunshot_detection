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


TESTING_RATIO = 0.7
SELF_RECORDING_WEIGHT = 50

#models
base_dir = "/home/gamagee/workspace/gunshot_detection/"
model_dir = base_dir+"raspberry_pi/models/"

sample_dir = base_dir+"REU_Data/spectrogram_training/samples_and_labels/"

augmented_labels_fn = sample_dir+"gunshot_augmented_sound_labels.npy"

augmented_samples_fn = sample_dir+"gunshot_augmented_sound_samples.npy"

spectrograph_samples_1_fn = sample_dir+"gunshot_augmented_sound_samples_spectro.npy"

spectrograph_samples_2_fn = sample_dir+"spectrogram_samples_power_to_db.npy"

augmented_labels = np.load(augmented_labels_fn)

augmented_samples = np.load(augmented_samples_fn)

spectrograph_samples_1 = np.load(spectrograph_samples_1_fn)

spectrograph_samples_2 = np.load(spectrograph_samples_2_fn)


sample_weights = np.array([1 for normally_recorded_sample in range(len(augmented_samples) - 660)] + [SELF_RECORDING_WEIGHT for raspberry_pi_recorded_sample in range(660)])
print("Shape of samples weights before splitting:", sample_weights.shape)

print("labels:",augmented_labels.shape)

print("samples_1:",augmented_samples.shape)

print("samples_2:",spectrograph_samples_1.shape)

print("samples_3:",spectrograph_samples_2.shape)

print("finished loading")

n = augmented_labels.shape[0]

print(n)

indexes = np.random.choice(n,int(n*TESTING_RATIO),replace=False)

print(indexes)

#samples
