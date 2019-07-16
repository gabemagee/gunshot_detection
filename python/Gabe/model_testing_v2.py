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


TESTING_RATIO = (2/3)
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

n = augmented_labels.shape[0]

print(n)

train_test_indexes = np.sort(np.random.choice(n,int(n*TESTING_RATIO),replace=False))
rrr = list(train_test_indexes)
print(len(rrr))
l = []
while len(l)<int(n*TESTING_RATIO/2):
    i = np.random.choice(len(rrr),1)
    print(i)
    l.append(rrr[i])
    del rrr[i]

#training

print(len(l))

print(len(rrr))

training_indexes = np.array(rrr)
np.save(sample_dir+"training_set_indexes.npy",training_indexes)

#TESTING

testing_indexes = np.array(l)
np.save(sample_dir+"testing_set_indexes.npy",testing_indexes)

#validation

#np.save(sample_dir+"training_set_indexes.npy",indexes)

print(len(indexes))

print(n*TESTING_RATIO)

#TESTING
exit()
labels_testing = []
samples_1_testing = []
samples_2_testing = []
samples_3_testing = []
weights_testing = []



#VALIDATION

labels_validation = []
samples_1_validation = []
samples_2_validation = []
samples_3_validation = []
weights_validation = []

print("about to loop")

for i in range(n):
    print(i)
    if i in indexes:
        print("in:",i)
        labels_testing.append(augmented_labels[i])
        samples_1_testing.append(augmented_samples[i])
        samples_2_testing.append(spectrograph_samples_1[i])
        samples_3_testing.append(spectrograph_samples_2[i])
        weights_testing.append(sample_weights[i])
    else:
        print("not:",i)
        labels_validation.append(augmented_labels[i])
        samples_1_validation.append(augmented_samples[i])
        samples_2_validation.append(spectrograph_samples_1[i])
        samples_3_validation.append(spectrograph_samples_2[i])
        weights_validation.append(sample_weights[i])




print("finished looping")

labels_testing = np.array(labels_testing)
samples_1_testing = np.array(samples_1_testing)
samples_2_testing = np.array(samples_2_testing)
samples_3_testing = np.array(samples_3_testing)
weights_testing = np.array(weights_testing)


labels_validation = np.array(labels_validation)
samples_1_validation = np.array(samples_1_validation)
samples_2_validation = np.array(samples_2_validation)
samples_3_validation = np.array(samples_3_validation)
weights_validation = np.array(weights_validation)

validation_dir = sample_dir+"validation/"
training_dir = sample_dir+"training/"

print("about to save")


np.save(training_dir+"labels.npy", labels_testing)
np.save(training_dir+"samples_1.npy", samples_1_testing)
np.save(training_dir+"samples_2.npy", samples_2_testing)
np.save(training_dir+"samples_3.npy", samples_3_testing)
np.save(training_dir+"weights.npy", weights_testing)

np.save(validation_dir+"labels.npy", labels_validation)
np.save(validation_dir+"samples_1.npy", samples_1_validation)
np.save(validation_dir+"samples_2.npy", samples_2_validation)
np.save(validation_dir+"samples_3.npy", samples_3_validation)
np.save(validation_dir+"weights.npy", weights_validation)
