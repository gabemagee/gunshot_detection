#!/usr/bin/env python
# coding: utf-8

# In[2]:


# File Directory 
import glob
import os
from os.path import isdir, join
from pathlib import Path

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
import librosa

# Dimension Reduction
from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

# Data Pre-processing
import pandas as pd
from sklearn.model_selection import KFold
import soundfile

# Deep Learning
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Configuration
py.init_notebook_mode(connected=True)

#%matplotlib inline


# In[7]:


# EDITED TO HAVE ONLY 2 LABELS: 1 for gunshot, 0 for everything else

samples=[]
sample_rates=[]
labels = []
sample_slice_iteration = 0
gunshot_aggregator = {}
glassbreak_aggregator = {}

gunshot_sound_dir = "/Users/laurenogden/school/iupui/gunshot/data/gunshot/"
# gunshot_sound_dir = "/home/lauogden/data/gunshot/"

for file in os.listdir(gunshot_sound_dir):
    if file.endswith(".wav"):
        #try:
        sample, sample_rate = librosa.load(gunshot_sound_dir + file)
        print("    read the file " + file)

        #add the 2 second long samples to the list of sample slices
        if len(sample) == 44100:
            samples.append(sample_slice)
            sample_rates.append(sample_rate)
            labels.append(label)
            print("    appended the slice")
            print("    size of samples is: " + str(len(samples)))

        #slice clips longer than 2 seconds long
        for i in range(0, sample.size - 44100, 44100):
            sample_slice = sample[i : i + 44100]
            print("    sliced the file " + file)
            label = 1
            gunshot_aggregator[sample_slice_iteration] = np.max(abs(sample_slice))
            sample_slice_iteration += 1
            if np.max(abs(sample_slice)) < 0.25:
                label = 0

            samples.append(sample_slice)
            sample_rates.append(sample_rate)
            labels.append(label)
            print("    appended the slice")
            print("    size of samples is: " + str(len(samples)))

        #except:
            #sample, sample_rate = soundfile.read(gunshot_sound_dir + file)
            #print("Gunshot sound unrecognized by Librosa:", sample)
            #pass
        
glassbreak_sound_dir = "/Users/laurenogden/school/iupui/gunshot/data/glassbreak/"
#glassbreak_sound_dir = "/home/lauogden/data/glassbreak/"

print("...Switching to glassbreak sounds...")

for file in os.listdir(glassbreak_sound_dir):
    if file.endswith(".wav"):
        try:
            sample, sample_rate = librosa.load(glassbreak_sound_dir + file)
            print("    read the file" + file)
            for i in range(0, sample.size - 44100, 44100):
                sample_slice = sample[i : i + 44100]
                print("    sliced the file" + file)
                label = 0
                glassbreak_aggregator[sample_slice_iteration] = np.max(abs(sample_slice))
                sample_slice_iteration += 1
                # if np.max(abs(sample_slice)) < 0.5:
                    # label = 0

                samples.append(sample_slice)
                sample_rates.append(sample_rate)
                labels.append(label)
                print("    appended the slice")
        except:
            sample, sample_rate = soundfile.read(glassbreak_sound_dir + file)
            print("Glassbreak sound unrecognized by Librosa:", sample)
            pass


# In[ ]:


#read in the csv file of descriptors for all other urban sounds
sound_types = pd.read_csv("/Users/laurenogden/school/iupui/gunshot/data/train/train.csv")
#sound_types = pd.read_csv("/home/lauogden/data/train/train.csv")
print(sound_types.loc[0,'Class'])



urban_aggregator = {}
j=0
#read in all of the wav files similar to above
urban_sound_dir = "/Users/laurenogden/school/iupui/gunshot/data/train/Train/"
#urban_sound_dir = "/home/lauogden/data/train/Train/"

print(os.listdir(urban_sound_dir))

for file in os.listdir(urban_sound_dir):
    if file.endswith(".wav"):
        try:
            sample, sample_rate = librosa.load(urban_sound_dir + file)
            print("    read the file" + file)
            for i in range(0, sample.size - 44100, 44100):
                sample_slice = sample[i : i + 44100]
                print("    sliced the file" + file)
                if(sound_types.loc[j, 'Class'] == "gun_shot"):
                    label = 1
                else:
                    label = 0
                urban_aggregator[sample_slice_iteration] = np.max(abs(sample_slice))
                sample_slice_iteration += 1
                if np.max(abs(sample_slice)) < 0.25:
                    label = 0

                samples.append(sample_slice)
                sample_rates.append(sample_rate)
                labels.append(label)
                print("    appended the slice")
            j +=1
        except:
            sample, sample_rate = soundfile.read(urban_sound_dir + file)
            print("Urban sound unrecognized by Librosa:", sample)
            pass


# In[23]:


'''
glassbreak_aggregator_inverted = dict([[v, k] for k, v in glassbreak_aggregator.items()])
for sl in sorted(glassbreak_aggregator.values(), reverse=True):
    print("Max value for slice #" + str(glassbreak_aggregator_inverted[sl]) + " is " + str(sl))
'''


# In[24]:


print("number of samples: ")
print(len(samples))
'''
i=744
samp=samples[i]
sr=sample_rates[i]
print(np.max(abs(samp)))
print(labels[i])
ipd.Audio(samp, rate=sr)
'''


# In[25]:


'''
def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)
'''


# In[26]:


'''
i=15
samp=samples[i]
sr=sample_rates[i]

freqs, times, spectrogram = log_specgram(samp, sr)

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + file)
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, 1,samp.size), samp)

ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + file)
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')
'''


# In[27]:


kf = KFold(n_splits=3, shuffle=True)
samples = np.array(samples)
labels = np.array(labels)
for train_index, test_index in kf.split(samples):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]


# In[29]:


# Parameters
lr = 0.001
generations = 20000
num_gens_to_wait = 250
batch_size = 256
drop_out_rate = 0.2
input_shape = (44100,1)


# In[30]:


#For Conv1D add Channel
train_wav = np.array(train_wav)
test_wav = np.array(test_wav)
train_wav = train_wav.reshape(-1,44100,1)
test_wav = test_wav.reshape(-1,44100,1)
train_label = keras.utils.to_categorical(train_label, 3)
test_label = keras.utils.to_categorical(test_label, 3)


# In[31]:


print(train_wav.shape)


# In[14]:


input_tensor = Input(shape=input_shape)

x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_tensor)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(4)(x)
x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(4)(x)
x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(6)(x)
x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(6)(x)
x = layers.Conv1D(256, 3, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(6)(x)
x = layers.Flatten()(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(50, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(20, activation='relu')(x)
output_tensor = layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(input_tensor, output_tensor)

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(lr = lr),
             metrics=['accuracy'])


# In[15]:


model.summary()


# In[ ]:


# add callbacks to stop early if it stops improving
model_filename = 'cnnmodel_w_urban.pkl' 
callbacks = [
    EarlyStopping(monitor='val_acc',
                  patience=10,
                  verbose=1,
                  mode='auto'),
    
    ModelCheckpoint(model_filename, monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    mode='auto'),
]


# In[16]:


model.fit(train_wav, train_label, 
          validation_data=[test_wav, test_label],
          batch_size=batch_size, 
          epochs=50,
          verbose=1)


# In[26]:


Y_test_pred = model.predict(test_wav)
y_predicted_classes_test = Y_test_pred.argmax(axis=-1)
y_actual_classes_test= test_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)


# In[18]:


print(wrong_examples)


# In[ ]:


# save the model
model.save("model_w_urban.h5")

