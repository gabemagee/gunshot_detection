#!/usr/bin/env python
# coding: utf-8

# In[1]:


### File Directory 
import glob
import os
from os.path import isdir, join
from pathlib import Path

### Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
import librosa

### Dimension Reduction
from sklearn.decomposition import PCA

### Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

### Data Pre-processing
import pandas as pd
from sklearn.model_selection import KFold
import soundfile

### Deep Learning
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

### Configuration
py.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Data Pre-Processing
samples=[]
sample_rates=[]
labels = []
sample_slice_iteration = 0
gunshot_aggregator = {}
glassbreak_aggregator = {}

### Acquiring gunshot sound data
gunshot_sound_dir = "/home/alexm/Datasets/gunshot_data/gunshot/"

print("...Parsing gunshot sounds...")

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
            print("Gunshot sound unrecognized by Librosa:", sample)
            pass

print("The number of samples of available for training is currently " + str(len(samples)) + '.')

### Acquiring sound data from examples of glass breaking
glassbreak_sound_dir = "/home/alexm/Datasets/gunshot_data/glassbreak/"

print("...Parsing sounds of glass breaking...")

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

print("The number of samples of available for training is currently " + str(len(samples)) + '.')


# In[3]:


### Reading in the CSV file of descriptors for all other kinds of urban sounds
sound_types = pd.read_csv("/home/alexm/Datasets/urban_sound_labels.csv")
urban_aggregator = {}
j=0

### Reading in all of the urban sound data WAV files
urban_sound_dir = "/home/alexm/Datasets/urban_sounds/"

for file in os.listdir(urban_sound_dir):
    if file.endswith(".wav"):
        try:
            sample, sample_rate = librosa.load(urban_sound_dir + file)
            for i in range(0, sample.size - 44100, 44100):
                sample_slice = sample[i : i + 44100]
                if(sound_types.loc[j, 'Class'] == "gun_shot"):
                    label = 2
                else:
                    label = 0
                urban_aggregator[sample_slice_iteration] = np.max(abs(sample_slice))
                sample_slice_iteration += 1
                if np.max(abs(sample_slice)) < 0.25:
                    label = 0

                samples.append(sample_slice)
                sample_rates.append(sample_rate)
                labels.append(label)
            j +=1
        except:
            sample, sample_rate = soundfile.read(urban_sound_dir + file)
            print("Urban sound not recognized by Librosa:", sample)
            pass

print("The number of samples of available for training is currently " + str(len(samples)) + '.')


# In[4]:


### Optional debugging after processing the data
i=0  # You can change the value of 'i' to adjust which sample is being inspected.
sample=samples[i]
sample_rate=sample_rates[i]
print("The number of samples available to the model for training is " + str(len(samples)) + '.')
print("The maximum frequency value in sample slice #" + str(i) + " is " + str(np.max(abs(sample))) + '.')
print("The label associated with sample slice #" + str(i) + " is " + str(labels[i]) + '.')
ipd.Audio(sample, rate=sample_rate)


# In[5]:


### Arranging the data
kf = KFold(n_splits=3, shuffle=True)
samples = np.array(samples)
labels = np.array(labels)
for train_index, test_index in kf.split(samples):
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]


# In[6]:


### Model parameters
learning_rate = 0.001
batch_size = 32
drop_out_rate = 0.2
input_shape = (44100,1)


# In[7]:


### Reshaping training & testing data
train_wav = train_wav.reshape(-1,44100,1)
test_wav = test_wav.reshape(-1,44100,1)
train_label = keras.utils.to_categorical(train_label, 3)
test_label = keras.utils.to_categorical(test_label, 3)


# In[8]:


### Optional debugging of the training data's shape
print(train_wav.shape)


# In[9]:


## Model Architecture
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
             optimizer=keras.optimizers.Adam(lr = learning_rate),
             metrics=['accuracy'])


# In[ ]:


### Configuring model properties
model_filename = 'gunshot_sound_model.pkl'

model_callbacks = [
    EarlyStopping(monitor='val_acc',
                  patience=10,
                  verbose=1,
                  mode='auto'),
    
    ModelCheckpoint(model_filename, monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    mode='auto'),
]

### Optional debugging of the model's architecture
model.summary()


# In[ ]:


## Model Training
training_generator = DataGenerator(train_wav, train_label)
validation_generator = DataGenerator(test_wav, test_label)

History = model.fit_generator(generator = training_generator,
                                validation_data = validation_generator,
                                epochs = 50,
                                callbacks = callbacks,
                                verbose = 1,
                                shuffle = True)


# In[ ]:


### Recording the model
model.load_weights("gunshot_sound_model.h5")
y_pred = np.round(model.predict(X_test))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
model.save_weights("model.h5")


# In[ ]:


### Optional debugging of incorrectly-labeled examples
Y_test_pred = model.predict(test_wav)
y_predicted_classes_test = Y_test_pred.argmax(axis=-1)
y_actual_classes_test= test_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)


# In[ ]:


### Optional debugging of an individual incorrectly-labeled example
i=0
samp=np.reshape(test_wav[i],44100,)
sr=sample_rates[i]
print(y_test[i],Y_test_pred[i])
ipd.Audio(samp, rate=sr)

