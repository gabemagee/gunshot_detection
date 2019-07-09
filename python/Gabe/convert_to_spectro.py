import os
import pandas as pd
import librosa
import librosa.display
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





def make_spectrogram(y):
    y = np.array(y)
    print(type(y))
    print(y.dtype)
    return np.array(librosa.feature.melspectrogram(y=y, sr=22050))

data_directory = "/home/gamagee/workspace/gunshot_detection/REU_Data/REU_Samples_and_Labels/"
label_csv = data_directory + "labels.csv"
sample_directory = data_directory + "Samples/"
base_dir = "/home/gamagee/workspace/gunshot_detection/REU_Data/spectrogram_training/"
sample_path = base_dir+"gunshot_augmented_sound_samples.npy"
label_path = base_dir+"gunshot_augmented_sound_labels.npy"
samples = np.load(sample_path)
labels = np.load(label_path)

n = len(samples)
c = 0

placeholder = []
for sample in samples:
    pct = (100*(c/n))
    if pct%2==0:
        print(pct,"%")
    a = make_spectrogram(sample)
    placeholder.append(a)
    c = c + 1
smpls = np.array(placeholder)
np.save(base_dir+"gunshot_augmented_sound_samples_spectro.npy",smpls)
print("done")
