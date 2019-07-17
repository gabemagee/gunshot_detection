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

from texttable import Texttable

SELF_RECORDING_WEIGHT = 50

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


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

model_list = []

name_dict = {}

models_dir = "/home/gamagee/workspace/gunshot_detection/REU_Data/spectrogram_training/models/"

model_filenames = os.listdir(models_dir)

CNN_2D_Model_keras = load_model(models_dir+"spectrogram_gunshot_model_1.h5")
name_dict[CNN_2D_Model_keras] = "CNN_2D_Model_keras"
model_list.append(CNN_2D_Model_keras)

CNN_1D_Model_keras = load_model(models_dir+"gunshot_sound_model_1d.h5",custom_objects={"auc":auc})
name_dict[CNN_1D_Model_keras] = "CNN_1D_Model_keras"
model_list.append(CNN_1D_Model_keras)


model_name = "SAME_INDEX_gunshot_2d_spectrogram_model.tflite"
gunshot_2d_spectrogram_model_tflite = tf.lite.Interpreter(models_dir+model_name)
gunshot_2d_spectrogram_model_tflite.allocate_tensors()
name_dict[gunshot_2d_spectrogram_model_tflite] = "gunshot_2d_spectrogram_model_tflite"
model_list.append(gunshot_2d_spectrogram_model_tflite)

model_name = "spectrogram_gunshot_model_1.tflite"
CNN_2D_Model_tflite = tf.lite.Interpreter(models_dir+model_name)
CNN_2D_Model_tflite.allocate_tensors()
name_dict[CNN_2D_Model_tflite] = "CNN_2D_Model_tflite"
model_list.append(CNN_2D_Model_tflite)


model_name = "gunshot_sound_model_1d.tflite"
CNN_1D_Model_tflite = tf.lite.Interpreter(models_dir+model_name)
CNN_1D_Model_tflite.allocate_tensors()
name_dict[CNN_1D_Model_tflite] = "CNN_1D_Model_tflite"
model_list.append(CNN_1D_Model_tflite)



def tflite_predict(interpreter,input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


print("loaded models")



def accuracy(true_pos,true_neg,false_pos,false_neg):
    return (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)

def precision(true_pos,true_neg,false_pos,false_neg):
    #TP/TP+FP
    return true_pos/(true_pos+false_pos)

def recall(true_pos,true_neg,false_pos,false_neg):
    #TP/TP+FN
    return true_pos/(true_pos+false_neg)

def f1_score(true_pos,true_neg,false_pos,false_neg):
    rc = recall(true_pos,true_neg,false_pos,false_neg)
    pr = precision(true_pos,true_neg,false_pos,false_neg)
    return 2*(rc * pr) / (rc + pr)

metrics = [accuracy,precision,recall,f1_score]

name_dict[accuracy] = "accuracy"
name_dict[precision] = "precision"
name_dict[recall] = "recall"
name_dict[f1_score] = "f1_score"

model_scores = {}
for model in model_list:
    model_scores[model] = {}
    for fig in ["true_pos","true_neg","false_pos","false_neg"]:
        model_scores[model][fig] = 0


predictions = []
for i in range(len(validation_wav)):
    print(i)
    x = validation_wav[i]
    print(x.shape)
    y = validation_label[i][0]
    d = {}
    for model in model_list:
        nm = name_dict[model]
        if nm.split("_")[-1]=="tflite":
            output = tflite_predict(model,x)
        else:
            print(model.layers[0].input_shape[0])
            x = x.reshape((-1, 128, 87, 1))
            output = model.predict(x)
        print(nm,y,output)
    #predictions.append(d)

exit()

t = Texttable()
table = []
table.append(["metric"]+model_list)
for metric in metrics:
    l = []
    #metric name
    l.append(name_dict[metric])
    for model in model_list:

        l.append(metric(model))
t.add_rows(table)
print(t.draw())