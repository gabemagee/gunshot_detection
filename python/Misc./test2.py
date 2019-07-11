import os
import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Deep learning
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

path = "data/gunshot"
files = os.listdir(path)
samples = []
sample_rates = []
labels = []
for filename in glob.glob(os.path.join(path, '*.wav')):
    try:
        file_names.append(filename)
        sample_rate, sample = wavfile.read(filename)
        sample = sample[:, 0]
        for i in xrange(0, len(sample) - 30000, 10000):
            samp0 = sample[i:i + 30000]
            lab = 2
            if (np.max(abs(samp0)) < 500.0):
                lab = 0
            samples.append(samp0)
            sample_rates.append(sample_rate)
            labels.append(lab)
    except:
        pass

path = "data/glassbreak"
files = os.listdir(path)
for filename in glob.glob(os.path.join(path, '*.wav')):
    try:
        file_names.append(filename)
        sample_rate, sample = wavfile.read(filename)
        sample = sample[:, 0]

        for i in xrange(0, len(sample) - 30000, 10000):
            samp0 = sample[i:i + 30000]
            lab = 1
            if (np.max(abs(samp0)) < 500.0):
                lab = 0

            samples.append(samp0)
            sample_rates.append(sample_rate)
            labels.append(lab)


    except:
        pass


def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


i = 15
samp = samples[i]
sr = sample_rates[i]

freqs, times, spectrogram = log_specgram(samp, sr)

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + filename)
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, 1, samp.size), samp)

ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + filename)
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')

i = 2
samp = samples[i]
sr = sample_rates[i]
print(labels[i])
ipd.Audio(samp, rate=sr)
train_wav, test_wav, train_label, test_label = train_test_split(wav_all, labels,
                                                                test_size=0.2,
                                                                random_state=1993,
                                                                shuffle=True)
# Parameters
lr = 0.001
generations = 20000
num_gens_to_wait = 250
batch_size = 512
drop_out_rate = 0.2
input_shape = (30000, 1)
# For Conv1D add Channel
train_wav = np.array(train_wav)
test_wav = np.array(test_wav)
train_wav = train_wav.reshape(-1, 30000, 1)
test_wav = test_wav.reshape(-1, 30000, 1)
train_label = keras.utils.to_categorical(train_label, 3)
test_label = keras.utils.to_categorical(test_label, 3)
print(train_wav.shape)
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
              optimizer=keras.optimizers.Adam(lr=lr),
              metrics=['accuracy'])
model.summary()

model.fit(train_wav, train_label, validation_data=[test_wav, test_label],
          batch_size=batch_size,
          epochs=50,
          verbose=1)

model.fit(train_wav, train_label, validation_data=[test_wav, test_label],
          batch_size=batch_size,
          epochs=50,
          verbose=1)

Y_test_pred = model.predict(test_wav)
y_classes = Y_test_pred.argmax(axis=-1)
y_test = test_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_classes != y_test)
print(wrong_examples)
i = 2
samp = np.reshape(test_wav[i], 30000, )
sr = sample_rates[i]
print(y_test[i], Y_test_pred[i])
ipd.Audio(samp, rate=sr)
i = 10
samp = np.reshape(test_wav[i], 30000, )
sr = sample_rates[i]
print(y_test[i], Y_test_pred[i])
ipd.Audio(samp, rate=sr)
i = 44
samp = np.reshape(test_wav[i], 30000, )
sr = sample_rates[i]
print(y_test[i], Y_test_pred[i])
ipd.Audio(samp, rate=sr)
i = 60
samp = np.reshape(test_wav[i], 30000, )
sr = sample_rates[i]
print(y_test[i], Y_test_pred[i])
ipd.Audio(samp, rate=sr)
i = 20
samp = np.reshape(test_wav[i], 30000, )
sr = sample_rates[i]
print(y_test[i], Y_test_pred[i])
ipd.Audio(samp, rate=sr)
