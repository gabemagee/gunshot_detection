import glob
import os
from os.path import isdir, join
from pathlib import Path


# ### Math Libraries

# In[ ]:


import numpy as np
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


# ### Data Pre-Processing Libraries

# In[ ]:


import pandas as pd
import librosa
import re
from sklearn.model_selection import KFold


# ### Visualization Libraries

# In[ ]:


import seaborn as sns
import IPython.display as ipd
import librosa.display


# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

print(os.getcwd())

number_of_desired_samples = 250

model_path = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_sound_model.h5"

labels = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_augmented_sound_labels.npy"

samples = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_augmented_sound_samples.npy"

model = keras.models.load_model(model_path,custom_objects={'auc':auc})
model.summary()


label_np = np.load(labels)
label_np = keras.utils.to_categorical(label_np, 2)
label_index = 0
gunshot_label_marker = 1
gunshot_indexes = []
non_gunshot_indexes = []
for label in label_np:
    if label[1]==1.0:
        gunshot_indexes.append(label_index)
    else:
        non_gunshot_indexes.append(label_index)
    label_index = label_index + 1
print(len(gunshot_indexes))
print(len(non_gunshot_indexes))

sub_sample_list_gs = []
sub_sample_list_uk = []
for i in range(number_of_desired_samples):
    r = np.random.randint(0,len(gunshot_indexes))
    while r in sub_sample_list_gs:
        r = np.random.randint(0,len(gunshot_indexes))
    sub_sample_list_gs.append(r)
for i in range(number_of_desired_samples):
    r = np.random.randint(0,len(non_gunshot_indexes))
    while r in sub_sample_list_uk:
        r = np.random.randint(0,len(non_gunshot_indexes))
    sub_sample_list_uk.append(r)

gunshot_samples = []
other_samples = []
sample_np = np.load(samples)
for index in sub_sample_list_gs:
    gunshot_samples.append(sample_np[index])
for index in sub_sample_list_uk:
    other_samples.append(sample_np[index])
sampling_rate_per_two_seconds = 44100

gunshot_samples = np.array(gunshot_samples)
gunshot_samples = gunshot_samples.reshape(-1, sampling_rate_per_two_seconds, 1)

other_samples = np.array(other_samples)
other_samples = other_samples.reshape(-1, sampling_rate_per_two_seconds, 1)

gunshots_correct = np.array((0,1)*number_of_desired_samples).reshape(-1,2,1)
other_correct = np.array((1,0)*number_of_desired_samples).reshape(-1,2,1)
<<<<<<< HEAD

print(gunshot_samples.shape)
print(gunshots_correct.shape)



gs_predictions = model.predict(gunshot_samples)
gs_predictions_classes = gs_predictions.argmax(axis=-1)
gs_actual_classes= gunshots_correct.argmax(axis=-1)
wrong_examples = np.nonzero(gs_predictions_classes != gs_actual_classes)
print(wrong_examples)
=======

print(gunshot_samples.shape)
print(gunshots_correct.shape)

loss, acc = model.evaluate(gunshot_samples, gunshots_correct)
print("Restored model, accuracy on gunshots: {:5.2f}%".format(100*acc))

loss, acc = model.evaluate(other_samples, other_correct)
print("Restored model, accuracy on other: {:5.2f}%".format(100*acc))
>>>>>>> ab11c1fea43fea7921c7abb94add785c2d98d475
