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
name_dict = {}

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def audio_to_melspectrogram(audio,hop_length=345*2):
    audio = np.array(audio)
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=44100,
                                                 n_mels=128,
                                                 hop_length=hop_length,
                                                 n_fft=128 * 20,
                                                 fmin=20,
                                                 fmax= 44100 // 2)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def make_spectrogram(y):
    y = np.array(y)
    return np.array(librosa.feature.melspectrogram(y=y, sr=22050))

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

def update_counts(y,output,model,model_scores):
    if y[0]=="gun_shot" and output[0]=="gun_shot":
        model_scores[model]["true_pos"] = model_scores[model]["true_pos"]+1
    elif y[0]=="gun_shot" and output[0]!="gun_shot":
        model_scores[model]["false_neg"] = model_scores[model]["false_neg"]+1
    elif y[0]!="gun_shot" and output[0]=="gun_shot":
        model_scores[model]["false_pos"] = model_scores[model]["false_pos"]+1
    elif y[0]!="gun_shot" and output[0]!="gun_shot":
        model_scores[model]["true_neg"] = model_scores[model]["true_neg"]+1

def tflite_predict(interpreter,input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #print(input_data.shape)
    input_data = np.array(input_data,dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

print("Available gpus:",get_available_gpus(),". Loading Data.")


base_dir = "/home/gamagee/workspace/gunshot_detection/"
sample_dir = base_dir+"REU_Data/spectrogram_training/samples_and_labels/"

sample_path = sample_dir+"gunshot_augmented_sound_samples.npy"
samples = np.load(sample_path)

sample_rate_per_two_seconds = 44100
number_of_classes = 2
sr = 22050
input_shape = (128, 87, 1)

label_path = sample_dir+"gunshot_augmented_sound_labels.npy"
labels = np.load(label_path)
labels = np.array([("gun_shot" if label ==1.0 else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels,1-labels))


#finding indexes

testing_indexes = np.load(base_dir+"raspberry_pi/indexes/testing_set_indexes.npy")
training_indexes = np.load(base_dir+"raspberry_pi/indexes/training_set_indexes.npy")

validation_wav = []
validation_label = []

for i in range(len(labels)):
    if i in training_indexes:
        pass
    elif i in testing_indexes:
        pass
    else:
        validation_wav.append(samples[i])
        validation_label.append(labels[i])

validation_wav = np.array(validation_wav)
validation_label = np.array(validation_label)

print("Finished loading data. Loading Models.")

model_list = []

tflite_model_list = []

models_dir = "/home/gamagee/workspace/gunshot_detection/REU_Data/spectrogram_training/models/"
models_dir = base_dir+"raspberry_pi/models/"

model_filenames = os.listdir(models_dir)

CNN_2D_keras = load_model(models_dir+"CNN_2D.h5")
name_dict[CNN_2D_keras] = "CNN_2D_keras"
model_list.append(CNN_2D_keras)

CNN_1D_keras = load_model(models_dir+"1D_CNN.h5",custom_objects={"auc":auc})
name_dict[CNN_1D_keras] = "CNN_1D_keras"
model_list.append(CNN_1D_keras)

gunshot_2d_spectrogram_model = load_model(models_dir+"RYAN_LATEST_gunshot_2d_spectrogram_model.h5",custom_objects={"auc":auc})
name_dict[gunshot_2d_spectrogram_model] = "gunshot_2d_spectrogram_model"
model_list.append(gunshot_2d_spectrogram_model)


CNN_2D_128x128_keras = load_model(models_dir+"128_128_gunshot_2d_spectrogram_model.h5",custom_objects={"auc":auc})
name_dict[CNN_2D_128x128_keras] = "CNN_2D_128x128_keras"
model_list.append(CNN_2D_128x128_keras)

model_name = "RYAN_LATEST_gunshot_2d_spectrogram_model.tflite"
gunshot_2d_spectrogram_model_tflite = tf.lite.Interpreter(models_dir+model_name)
gunshot_2d_spectrogram_model_tflite.allocate_tensors()
name_dict[gunshot_2d_spectrogram_model_tflite] = "gunshot_2d_spectrogram_model_tflite"
tflite_model_list.append(gunshot_2d_spectrogram_model_tflite)

model_name = "CNN_2D.tflite"
CNN_2D_Model_tflite = tf.lite.Interpreter(models_dir+model_name)
CNN_2D_Model_tflite.allocate_tensors()
name_dict[CNN_2D_Model_tflite] = "CNN_2D_Model_tflite"
tflite_model_list.append(CNN_2D_Model_tflite)


model_name = "1D_CNN.tflite"
CNN_1D_Model_tflite = tf.lite.Interpreter(models_dir+model_name)
CNN_1D_Model_tflite.allocate_tensors()
name_dict[CNN_1D_Model_tflite] = "CNN_1D_Model_tflite"
tflite_model_list.append(CNN_1D_Model_tflite)

model_name = "128_128_gunshot_2d_spectrogram_model.tflite"
CNN_2D_128x128_tflite = tf.lite.Interpreter(models_dir+model_name)
CNN_2D_128x128_tflite.allocate_tensors()
name_dict[CNN_2D_128x128_tflite] = "CNN_2D_128x128_tflite"
tflite_model_list.append(CNN_2D_128x128_tflite)


one_and_two  = tf.keras.Model()
one_and_three = tf.keras.Model()
two_and_three = tf.keras.Model()
one_and_four = tf.keras.Model()
two_and_four = tf.keras.Model()
three_and_four = tf.keras.Model()
model_list.append(one_and_two)
model_list.append(one_and_three)
model_list.append(two_and_three)
model_list.append(one_and_four)
model_list.append(two_and_four)
model_list.append(three_and_four)
name_dict[one_and_two] = "1 and 2"
name_dict[one_and_three] = "1 and 3"
name_dict[two_and_three] = "2 and 3"
name_dict[one_and_four] = "one_and_four"
name_dict[two_and_four] = "two_and_four"
name_dict[three_and_four] = "three_and_four"

model_scores = {}
for model in model_list:
    model_scores[model] = {}
    for fig in ["true_pos","true_neg","false_pos","false_neg"]:
        model_scores[model][fig] = 0

for model in tflite_model_list:
    model_scores[model] = {}
    for fig in ["true_pos","true_neg","false_pos","false_neg"]:
        model_scores[model][fig] = 0

print("loaded models")

metrics = [accuracy,precision,recall,f1_score]

name_dict[accuracy] = "accuracy"
name_dict[precision] = "precision"
name_dict[recall] = "recall"
name_dict[f1_score] = "f1_score"

last = 0
for i in range(len(validation_wav)):
    temp = int(i*100/len(validation_wav))
    if temp> last:
        last = temp
        print(last)
    x = validation_wav[i]
    #print(x.shape)
    y = label_binarizer.inverse_transform(validation_label[:,0][i])

    #CNN_2D_keras
    x_1 = make_spectrogram(x).reshape((-1, 128, 87, 1))
    #print("input shape",x_1.shape)
    #print("model input shape",CNN_2D_Model_keras.layers[0].input_shape[0])
    output = CNN_2D_keras.predict(x_1)[:,0][0]
    output = label_binarizer.inverse_transform(output)
    update_counts(y,output,CNN_2D_keras,model_scores)
    output_1 = output


    #CNN_1D_Model_keras
    x_1 = x.reshape((-1, 44100, 1))
    #print("input shape",x_1.shape)
    #print("model input shape",CNN_1D_Model_keras.layers[0].input_shape[0])
    output = CNN_1D_keras.predict(x_1)[:,0][0]
    output = label_binarizer.inverse_transform(output)
    update_counts(y,output,CNN_1D_keras,model_scores)
    output_2 = output

    #gunshot_2d_spectrogram_model
    x_1 = audio_to_melspectrogram(x).reshape((-1,128,64,1))
    #print("input shape",x_1.shape)
    #print("model input shape",gunshot_2d_spectrogram_model.layers[0].input_shape)
    output = gunshot_2d_spectrogram_model.predict(x_1)[:,0][0]
    output = label_binarizer.inverse_transform(output)
    update_counts(y,output,gunshot_2d_spectrogram_model,model_scores)
    output_3 = output

    #CNN_2D_128x128_keras
    x_1 = audio_to_melspectrogram(x,hop_length=345).reshape((-1,128,128,1))
    output = CNN_2D_128x128_keras.predict(x_1)[:,0][0]
    output = label_binarizer.inverse_transform(output)
    update_counts(y,output,CNN_2D_128x128_keras,model_scores)
    output_4 = output

    # one_and_two
    if output_1[0]=="gun_shot" and output_2[0]=="gun_shot":
        output=["gun_shot"]
    else:
        output = ["other"]
    update_counts(y,output,one_and_two,model_scores)

    # two_and_three
    if output_2[0]=="gun_shot" and output_3[0]=="gun_shot":
        output=["gun_shot"]
    else:
        output = ["other"]
    update_counts(y,output,two_and_three,model_scores)

    # one_and_three
    if output_1[0]=="gun_shot" and output_3[0]=="gun_shot":
        output=["gun_shot"]
    else:
        output = ["other"]
    update_counts(y,output,one_and_three,model_scores)

    # one_and_four
    if output_1[0]=="gun_shot" and output_4[0]=="gun_shot":
        output=["gun_shot"]
    else:
        output = ["other"]
    update_counts(y,output,one_and_four,model_scores)

    # two_and_four
    if output_2[0]=="gun_shot" and output_4[0]=="gun_shot":
        output=["gun_shot"]
    else:
        output = ["other"]
    update_counts(y,output,two_and_four,model_scores)

    # three_and_four
    if output_3[0]=="gun_shot" and output_4[0]=="gun_shot":
        output=["gun_shot"]
    else:
        output = ["other"]
    update_counts(y,output,three_and_four,model_scores)


    ## TFLite
    interpreter = gunshot_2d_spectrogram_model_tflite
    x_1 = audio_to_melspectrogram(x).reshape((-1,128,64,1))
    output = tflite_predict(interpreter,x_1)
    print(output)

    interpreter = CNN_2D_Model_tflite
    x_1 = make_spectrogram(x).reshape((-1, 128, 87, 1))
    output = tflite_predict(interpreter,x_1)
    print(output)


    interpreter = CNN_1D_Model_tflite
    shape = interpreter.get_input_details()[0]['shape']
    print(shape)
    x_1 = x.reshape((-1, 44100, 1))
    print(x_1.shape)
    output = tflite_predict(interpreter,x_1)
    print(output)

    interpreter = CNN_2D_128x128_tflite
    x_1 = audio_to_melspectrogram(x,hop_length=345).reshape((-1,128,128,1))
    output = tflite_predict(interpreter,x_1)
    print(output)



t = Texttable()
table = []
table.append(["metric"]+[name_dict[model] for model in model_list])
for metric in metrics:
    l = []
    #metric name
    l.append(name_dict[metric])
    for model in model_list:
        [model_scores[model]["true_pos"],]
        l.append(metric(model_scores[model]["true_pos"],model_scores[model]["true_neg"],model_scores[model]["false_pos"],model_scores[model]["false_neg"]))
    table.append(l)
t.add_rows(table)
print(t.draw())
