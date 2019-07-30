import os
import glob
import csv
import IPython
import tensorflow as tf
import tensorflow.keras as keras
import IPython.display as ipd
import numpy as np
import sys
import keras
import librosa
import progressbar
import pickle
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalAveragePooling1D, MaxPooling1D, Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras.optimizers import Adam, SGD
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.client import device_lib
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from texttable import Texttable



name_dict = {}
model_list = []
model_dict = {}

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

def prep_model(file_path):
    model = load_model(file_path,custom_objects={"auc":auc})
    name_dict[model] = file_path.split("/")[-1].split(".")[0]
    model_list.append(model)
    model_dict[name_dict[model]] = model

def IOU(true_pos,true_neg,false_pos,false_neg):
    denom = true_pos+false_pos+false_neg
    return true_pos/denom



def update_counts(y,output,model,model_scores):
    #print(name_dict[model],model_scores[model])
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



data_dir = "/home/gamagee/workspace/gunshot_detection/REU_Data/ryan_model/data/"
validation_wav = np.load(data_dir+"augmented_validation_samples.npy")
labels = np.load(data_dir+"augmented_validation_labels.npy")
sample_rate_per_two_seconds = 44100
number_of_classes = 2
sr = 22050
input_shape = (128, 87, 1)

print(labels)

print("----")
labels = np.array([("gun_shot" if label =="gun_shot" else "other") for label in labels])

print(labels)

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
validation_label = np.hstack((labels,1-labels))

print(validation_label.shape)
print(validation_label)
for mdl in model_list:
    print(name_dict[mdl])

print("Finished loading data. Loading Models.")



models_dir = "/home/gamagee/workspace/gunshot_detection/REU_Data/ryan_model/models/"

for model_filename in os.listdir(models_dir):
    prep_model(models_dir+model_filename)

to_append = []


from itertools import combinations 

for model_1,model_2 in combinations(model_list, 2) :
    model_1_name = name_dict[model_1]
    model_2_name = name_dict[model_2]
    and_model_name = model_1_name+"_and_"+model_2_name
    or_model_name = model_1_name+"_or_"+model_2_name
    and_model = tf.keras.Model()
    or_model = tf.keras.Model()
    to_append.append(and_model)
    to_append.append(or_model)
    name_dict[and_model] = and_model_name
    name_dict[or_model] = or_model_name
    model_dict[and_model_name] = and_model
    model_dict[or_model_name] = or_model


model_list.extend(to_append)


majority = tf.keras.Model()
name_dict[majority] = "majority"
model_dict["majority"] = majority
model_list.append(majority)

for mdl in model_list:
    print(name_dict[mdl])



model_scores = {}
for model in model_list:
    model_scores[model] = {}
    for fig in ["true_pos","true_neg","false_pos","false_neg"]:
        model_scores[model][fig] = 0

print("loaded models")

metrics = [accuracy,precision,recall,f1_score,IOU]

name_dict[accuracy] = "accuracy"
name_dict[precision] = "precision"
name_dict[recall] = "recall"
name_dict[f1_score] = "f1_score"
name_dict[IOU] = "IOU"

last = 0
bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
bar.update(last)

print(validation_label[:,0])

for i in range(len(validation_wav)):
    temp = int(i*100/len(validation_wav))
    if temp> last:
        last = temp
        bar.update(last)
    x = validation_wav[i]
    y = label_binarizer.inverse_transform(validation_label[:,0][i])


    # 1D
    x_1 = x.reshape((-1, 44100, 1))
    model = model_dict["1_dimensional"]
    output = model.predict(x_1)[:,0][0]
    output_1 = label_binarizer.inverse_transform(output)
    update_counts(y,output_1,model,model_scores)

    # 128x128
    x_1 = audio_to_melspectrogram(x,hop_length=345).reshape((-1,128,128,1))
    model = model_dict["128_x_128"]
    output = model.predict(x_1)[:,0][0]
    output_2 = label_binarizer.inverse_transform(output)
    update_counts(y,output_2,model,model_scores)

    # 128x64
    x_1 = audio_to_melspectrogram(x).reshape((-1,128,64,1))
    model = model_dict["128_x_64"]
    output = model.predict(x_1)[:,0][0]
    output_3 = label_binarizer.inverse_transform(output)
    update_counts(y,output_3,model,model_scores)


    #OR
    #1 2
    model = model_dict["128_x_128_or_1_dimensional"]
    if (output_1 =="gun_shot" or output_2 =="gun_shot"):
        output = "gun_shot"
    else:
        output = "other"
    update_counts(y,output,model,model_scores)

    #1 3
    model = model_dict["128_x_64_or_1_dimensional"]
    if (output_1 =="gun_shot" or output_3 =="gun_shot"):
        output = "gun_shot"
    else:
        output = "other"
    update_counts(y,output,model,model_scores)

    #2 3
    model = model_dict["128_x_64_or_128_x_128"]
    if (output_2 =="gun_shot" or output_3 =="gun_shot"):
        output = "gun_shot"
    else:
        output = "other"
    update_counts(y,output,model,model_scores)

    #AND

    #1 2
    model = model_dict["128_x_128_and_1_dimensional"]
    if (output_1 =="gun_shot" and output_2 =="gun_shot"):
        output = "gun_shot"
    else:
        output = "other"
    update_counts(y,output,model,model_scores)

    #1 3
    model = model_dict["128_x_64_and_1_dimensional"]
    if (output_1 =="gun_shot" and output_3 =="gun_shot"):
        output = "gun_shot"
    else:
        output = "other"
    update_counts(y,output,model,model_scores)

    #2 3
    model = model_dict["128_x_64_and_128_x_128"]
    if (output_2 =="gun_shot" and output_3 =="gun_shot"):
        output = "gun_shot"
    else:
        output = "other"
    update_counts(y,output,model,model_scores)


    #MAJORITY
    model = model_dict["majority"]
    sum = 0
    for boolean_expression in [output_1,output_2,output_3]:
        if boolean_expression=="gunshot":
            sum = sum + 1
    if sum>1:
        output = "gun_shot"
    else:
        output = "other"
    update_counts(y,output,model,model_scores)



bar.finish()

t = Texttable()
table = []
table.append(["metric"]+[name_dict[model] for model in model_list])
for metric in metrics:
    l = []
    #metric name
    l.append(name_dict[metric])
    for model in model_list:
        l.append(metric(model_scores[model]["true_pos"],model_scores[model]["true_neg"],model_scores[model]["false_pos"],model_scores[model]["false_neg"]))
    table.append(l)
t.add_rows(table)
print(t.draw())
