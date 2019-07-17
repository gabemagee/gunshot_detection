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
import librosa
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

sample_path = sample_dir+"gunshot_augmented_sound_samples.npy"

#sample_path = sample_dir+"gunshot_augmented_sound_samples_spectro.npy"

spectrograph_samples_2_fn = sample_dir+"spectrogram_samples_power_to_db.npy"

samples = np.load(sample_path)
labels = np.load(label_path)

sample_weights = np.array([1 for normally_recorded_sample in range(len(samples) - 660)] + [SELF_RECORDING_WEIGHT for raspberry_pi_recorded_sample in range(660)])


print(samples.shape)

#samples.reshape(-1,128,87,1)
sample_rate_per_two_seconds = 44100
number_of_classes = 2
sr = 22050
input_shape = (128, 87, 1)

print(labels.shape)



print(list(labels))
labels = np.array([("gun_shot" if label ==1.0 else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels,1-labels))
print(labels)

print(label_binarizer.inverse_transform(labels[:,0]))

sampling_rate = 44100
hop_length = 345 * 2
fmin = 20
fmax = sampling_rate // 2
n_mels = 128
n_fft = n_mels * 20
min_seconds = 0.5
def audio_to_melspectrogram(audio):
    audio = np.array(audio)
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=sampling_rate,
                                                 n_mels=n_mels,
                                                 hop_length=hop_length,
                                                 n_fft=n_fft,
                                                 fmin=fmin,
                                                 fmax=fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


testing_indexes_path = base_dir+"raspberry_pi/indexes/testing_set_indexes.npy"

testing_indexes = np.load(testing_indexes_path)

training_indexes_path = base_dir+"raspberry_pi/indexes/training_set_indexes.npy"

training_indexes = np.load(training_indexes_path)



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




gunshot_2d_spectrogram_model = load_model(models_dir+"RYAN_LATEST_gunshot_2d_spectrogram_model.h5",custom_objects={"auc":auc})
name_dict[gunshot_2d_spectrogram_model] = "gunshot_2d_spectrogram_model"
model_list.append(gunshot_2d_spectrogram_model)


model_name = "SAME_INDEX_gunshot_2d_spectrogram_model.tflite"
gunshot_2d_spectrogram_model_tflite = tf.lite.Interpreter(models_dir+model_name)
gunshot_2d_spectrogram_model_tflite.allocate_tensors()
name_dict[gunshot_2d_spectrogram_model_tflite] = "gunshot_2d_spectrogram_model_tflite"
#model_list.append(gunshot_2d_spectrogram_model_tflite)

model_name = "spectrogram_gunshot_model_1.tflite"
CNN_2D_Model_tflite = tf.lite.Interpreter(models_dir+model_name)
CNN_2D_Model_tflite.allocate_tensors()
name_dict[CNN_2D_Model_tflite] = "CNN_2D_Model_tflite"
#model_list.append(CNN_2D_Model_tflite)


model_name = "gunshot_sound_model_1d.tflite"
CNN_1D_Model_tflite = tf.lite.Interpreter(models_dir+model_name)
CNN_1D_Model_tflite.allocate_tensors()
name_dict[CNN_1D_Model_tflite] = "CNN_1D_Model_tflite"
#model_list.append(CNN_1D_Model_tflite)



def tflite_predict(interpreter,input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def make_spectrogram(y):
    y = np.array(y)
    return np.array(librosa.feature.melspectrogram(y=y, sr=22050))


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
    #print(i)
    x = validation_wav[i]
    #print(x.shape)
    y = label_binarizer.inverse_transform(validation_label[:,0][i])

    #CNN_2D_Model_keras
    x_1 = make_spectrogram(x).reshape((-1, 128, 87, 1))
    #print("input shape",x_1.shape)
    #print("model input shape",CNN_2D_Model_keras.layers[0].input_shape[0])
    output = CNN_2D_Model_keras.predict(x_1)[:,0][0]
    output = label_binarizer.inverse_transform(output)
    if y[0]=="gun_shot" and output[0]=="gun_shot":
        model_scores[CNN_2D_Model_keras]["true_pos"] = model_scores[CNN_2D_Model_keras]["true_pos"]+1
    elif y[0]=="gun_shot" and output[0]!="gun_shot":
        model_scores[CNN_2D_Model_keras]["false_neg"] = model_scores[CNN_2D_Model_keras]["false_neg"]+1
    elif y[0]!="gun_shot" and output[0]=="gun_shot":
        model_scores[CNN_2D_Model_keras]["false_pos"] = model_scores[CNN_2D_Model_keras]["false_pos"]+1
    elif y[0]!="gun_shot" and output[0]!="gun_shot":
        model_scores[CNN_2D_Model_keras]["true_neg"] = model_scores[CNN_2D_Model_keras]["true_neg"]+1


    #CNN_1D_Model_keras
    x_1 = x.reshape((-1, 44100, 1))
    #print("input shape",x_1.shape)
    #print("model input shape",CNN_1D_Model_keras.layers[0].input_shape[0])
    output = CNN_1D_Model_keras.predict(x_1)[:,0][0]
    output = label_binarizer.inverse_transform(output)
    if y[0]=="gun_shot" and output[0]=="gun_shot":
        model_scores[CNN_1D_Model_keras]["true_pos"] = model_scores[CNN_1D_Model_keras]["true_pos"]+1
    elif y[0]=="gun_shot" and output[0]!="gun_shot":
        model_scores[CNN_1D_Model_keras]["false_neg"] = model_scores[CNN_1D_Model_keras]["false_neg"]+1
    elif y[0]!="gun_shot" and output[0]=="gun_shot":
        model_scores[CNN_1D_Model_keras]["false_pos"] = model_scores[CNN_1D_Model_keras]["false_pos"]+1
    elif y[0]!="gun_shot" and output[0]!="gun_shot":
        model_scores[CNN_1D_Model_keras]["true_neg"] = model_scores[CNN_1D_Model_keras]["true_neg"]+1

    #gunshot_2d_spectrogram_model
    x_1 = audio_to_melspectrogram(x).reshape((-1,128,64,1))
    #print("input shape",x_1.shape)
    #print("model input shape",gunshot_2d_spectrogram_model.layers[0].input_shape)
    output = gunshot_2d_spectrogram_model.predict(x_1)[:,0][0]
    output = label_binarizer.inverse_transform(output)
    if y[0]=="gun_shot" and output[0]=="gun_shot":
        model_scores[gunshot_2d_spectrogram_model]["true_pos"] = model_scores[gunshot_2d_spectrogram_model]["true_pos"]+1
    elif y[0]=="gun_shot" and output[0]!="gun_shot":
        model_scores[gunshot_2d_spectrogram_model]["false_neg"] = model_scores[gunshot_2d_spectrogram_model]["false_neg"]+1
    elif y[0]!="gun_shot" and output[0]=="gun_shot":
        model_scores[gunshot_2d_spectrogram_model]["false_pos"] = model_scores[gunshot_2d_spectrogram_model]["false_pos"]+1
    elif y[0]!="gun_shot" and output[0]!="gun_shot":
        model_scores[gunshot_2d_spectrogram_model]["true_neg"] = model_scores[gunshot_2d_spectrogram_model]["true_neg"]+1

    for model in model_list:
        print(name_dict[model])
        for fig in ["true_pos","true_neg","false_pos","false_neg"]:
            print(fig,model_scores[model][fig])

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
