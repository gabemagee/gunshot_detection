#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import IPython.display as ipd

# Data Pre-processing
import pandas as pd
import soundfile
from sklearn.model_selection import KFold

# Deep Learning
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K


# In[2]:


samples=[]
sample_rates=[]
labels = []
sample_slice_iteration = 0
gunshot_aggregator = {}
glassbreak_aggregator = {}

gunshot_sound_dir = "~/Datasets/gunshot_data/gunshot/"

for file in os.listdir(gunshot_sound_dir):
    if file.endswith(".wav"):
        try:
            sample, sample_rate = librosa.load(gunshot_sound_dir + file)
            for i in range(0, sample.size - 44100, 44100):
                sample_slice = sample[i : i + 44100]
                label = 2
                gunshot_aggregator[sample_slice_iteration] = np.max(abs(sample_slice))
                sample_slice_iteration += 1
                if np.max(abs(sample_slice)) < 0.25:
                    label = 0

                samples.append(sample_slice)
                sample_rates.append(sample_rate)
                labels.append(label)
        except:
            sample, sample_rate = soundfile.read(gunshot_sound_dir + file)
            #print("Gunshot sound unrecognized by Librosa:", sample)
            pass
        
gunshot_sound_dir = "~/Datasets/gunshot_data/glassbreak/"

print("...Switching to glassbreak sounds...")

for file in os.listdir(glassbreak_sound_dir):
    if file.endswith(".wav"):
        try:
            sample, sample_rate = librosa.load(glassbreak_sound_dir + file)
            for i in range(0, sample.size - 44100, 44100):
                sample_slice = sample[i : i + 44100]
                label = 1
                glassbreak_aggregator[sample_slice_iteration] = np.max(abs(sample_slice))
                sample_slice_iteration += 1
                if np.max(abs(sample_slice)) < 0.5:
                    label = 0

                samples.append(sample_slice)
                sample_rates.append(sample_rate)
                labels.append(label)
        except:
            sample, sample_rate = soundfile.read(glassbreak_sound_dir + file)
            print("Glassbreak sound unrecognized by Librosa:", sample)
            pass


# In[3]:


glassbreak_aggregator_inverted = dict([[v, k] for k, v in glassbreak_aggregator.items()])
for sl in sorted(glassbreak_aggregator.values(), reverse=True):
    print("Max value for slice #" + str(glassbreak_aggregator_inverted[sl]) + " is " + str(sl))


# In[4]:


print(len(samples))
i=744
samp=samples[i]
sr=sample_rates[i]
print(np.max(abs(samp)))
print(labels[i])
ipd.Audio(samp, rate=sr)


# In[5]:


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


# In[6]:


i=15
samp=samples[i]
sr=sample_rates[i]

freqs, times, spectrogram = log_specgram(samp, sr)

# fig = plt.figure(figsize=(14, 8))
# ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + file)
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, 1,samp.size), samp)

# ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + file)
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')


# In[7]:


kf = KFold(n_splits=3, shuffle=True)
samples = np.array(samples)
labels = np.array(labels)
for train_index, test_index in kf.split(samples):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]


# In[10]:


# Parameters
lr = 0.001
generations = 20000
num_gens_to_wait = 250
batch_size = 256
drop_out_rate = 0.2
input_shape = (44100,1)


# In[11]:


#For Conv1D add Channel
train_wav = np.array(train_wav)
test_wav = np.array(test_wav)
train_wav = train_wav.reshape(-1,44100,1)
test_wav = test_wav.reshape(-1,44100,1)
train_label = keras.utils.to_categorical(train_label, 3)
test_label = keras.utils.to_categorical(test_label, 3)


# In[13]:


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


# In[19]:


i=1
samp=np.reshape(test_wav[i],44100,)
sr=sample_rates[i]
print(y_test[i],Y_test_pred[i])
ipd.Audio(samp, rate=sr)


# In[ ]:


i=5
samp=np.reshape(test_wav[i],44100,)
sr=sample_rates[i]
print(y_test[i],Y_test_pred[i])
ipd.Audio(samp, rate=sr)


# In[17]:


i=19
samp=np.reshape(test_wav[i],44100,)
sr=sample_rates[i]
print(y_test[i],Y_test_pred[i])
ipd.Audio(samp, rate=sr)


# In[18]:


i=41
samp=np.reshape(test_wav[i],44100,)
sr=sample_rates[i]
print(y_test[i],Y_test_pred[i])
ipd.Audio(samp, rate=sr)


# In[19]:


i=50
samp=np.reshape(test_wav[i],44100,)
sr=sample_rates[i]
print(y_test[i],Y_test_pred[i])
ipd.Audio(samp, rate=sr)


# In[ ]:





# In[ ]:




