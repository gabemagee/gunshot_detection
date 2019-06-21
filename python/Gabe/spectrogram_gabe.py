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

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
# In[8]:


def make_spectrogram(y,sr):
    return np.array(librosa.feature.melspectrogram(y=y, sr=sr))


def get_categories():
    s = []
    d = {}
    with open(label_csv,"r") as lblcsv:
        c = list(csv.reader(lblcsv))
        header = c[0]
        for row in c[1:]:
            e = {}
            e["label"] = row[1]
            e["source"] = row[2]
            d[row[0]+".wav"] = e
            if row[1] not in s:
                s.append(row[1])
    return s,d


#ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

data_directory = "/home/gamagee/workspace/gunshot_detection/REU_Data/REU_Samples_and_Labels/"
label_csv = data_directory + "labels.csv"
sample_directory = data_directory + "Samples/"
base_dir = "/home/gamagee/workspace/gunshot_detection/REU_Data/"
sample_path = base_dir+"gabe_sample.npy"
label_path = base_dir+"gabe_label.npy"
samples = np.load(sample_path)
labels = np.load(label_path)
samples.reshape(-1,128,87,1)
sample_rate_per_two_seconds = 44100
number_of_classes = 2
sr = 22050
input_shape = (128, 87, 1)
kf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in kf.split(samples):
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]


def model(train_wav, train_label, test_label, test_wav, name,verbose=1,drop_out_rate = 0.1,learning_rate = 0.01,number_of_epochs = 100,batch_size = 32,filter_size = (4,4),maxpool_size = (2,2),activation = "relu"):
    optimizer = optimizers.Adam(learning_rate, learning_rate / 100)
    input_tensor = Input(shape=input_shape)
    metrics = [auc, "accuracy"]
    #Model Architecture
    x = layers.Conv2D(16, filter_size, activation=activation, padding="same")(input_tensor)
    x = layers.Conv2D(16, filter_size, activation=activation, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(maxpool_size)(x)
    x = layers.Dropout(rate=drop_out_rate)(x)

    x = layers.Conv2D(32, filter_size, activation=activation, padding="same")(x)
    x = layers.Conv2D(32, filter_size, activation=activation, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(maxpool_size)(x)
    x = layers.Dropout(rate=drop_out_rate)(x)

    x = layers.Conv2D(64, filter_size, activation=activation, padding="same")(x)
    x = layers.Conv2D(64, filter_size, activation=activation, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(maxpool_size)(x)
    x = layers.Dropout(rate=drop_out_rate)(x)

    x = layers.Conv2D(256, filter_size, activation=activation, padding="same")(x)
    x = layers.Conv2D(256, filter_size, activation=activation, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Dropout(rate=(drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting

    x = layers.Dense(64, activation=activation)(x)
    x = layers.Dense(1028, activation=activation)(x)
    output_tensor = layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(input_tensor, output_tensor)
    model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=metrics)

    #Configuring model properties
    model_filename = base_dir + "gunshot_sound_model_spectrograph_"+name+".pkl"

    model_callbacks = [
        EarlyStopping(monitor='val_acc',
                      patience=10,
                      verbose=verbose,
                      mode='auto'),

        ModelCheckpoint(model_filename, monitor='val_acc',
                        verbose=verbose,
                        save_best_only=True,
                        mode='auto'),
    ]
    #Optional debugging of the model's architecture
    model.summary()

    test_wav = test_wav.reshape(-1,128,87,1)
    train_wav = train_wav.reshape(-1,128, 87, 1)

    #Training & caching the model
    History = model.fit(train_wav, train_label,
              validation_data=[test_wav, test_label],
              epochs=number_of_epochs,
              callbacks=model_callbacks,
              verbose=verbose,
              batch_size=batch_size,
              shuffle=True)
    model.save(base_dir + "gunshot_sound_model_spectrograph_"+name+".h5")
    return model.evaluate(test_wav, test_label, batch_size=batch_size)

drop_out_rates = 0.1,0.05,0.01,0.25
learning_rates = 0.1,0.05,0.01
filter_sizes = (4,4),(5,5),(6,6),(3,3)
name = "model"
print(model(train_wav, train_label, test_label, test_wav, name= name,drop_out_rate=0.1,learning_rate=0.01,filter_size=(3,3),verbose=1))


"""
norm_samples = np.load(base_dir + "gunshot_sound_samples.npy")
norm_labels = np.load(base_dir + "gunshot_sound_labels.npy")

aug_samples = np.load(base_dir + "gunshot_augmented_sound_samples.npy")
aug_labels = np.load(base_dir + "gunshot_augmented_sound_labels.npy")

labels = np.concatenate((aug_labels,norm_labels))
samples = np.concatenate((aug_samples,norm_samples))

labels = keras.utils.to_categorical(labels, 2)

print(labels.shape)
print(samples.shape)

eee = 0
for file in os.listdir(sample_directory):
    print(eee)
    eee = eee +1
    sample,sr = librosa.load(sample_directory+file)
    if len(sample) <= sample_rate_per_two_seconds:
        number_of_missing_frames = sample_rate_per_two_seconds - len(sample)
        padded_sample = np.array(sample.tolist() + [0 for i in range(number_of_missing_frames)])
        label = d[file]["label"]
        samples.append(padded_sample)
        labels.append(label)
        ids.append(file.split(".")[0])
    else:
        number_of_missing_frames = len(sample) % sample_rate_per_two_seconds
        sample = np.array(sample.tolist() + [0 for i in range(number_of_missing_frames)])
        for i in range(0, sample.size - sample_rate_per_two_seconds, sample_rate_per_two_seconds):
            sample_slice = sample[i : i + sample_rate_per_two_seconds]
            label = d[file]["label"]
            if label == "gun_shot":
                labels.append(1)
            else:
                labels.append(0)
            samples.append(sample_slice)

            ids.append(file.split(".")[0])

sa = []
for sample in samples:
    a = make_spectrogram(sample,sr)
    sa.append(a)
samples = np.array(sa).reshape(input_shape)

sample_path = base_dir+"gabe_sample.npy"
label_path = base_dir+"gabe_label.npy"

np.save(sample_path,samples)
np.save(label_path,labels)
"""
