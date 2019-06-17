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

number_of_desired_samples = 250
sampling_rate_per_two_seconds = 44100

model_path = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_sound_model.h5"

labels = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_augmented_sound_labels.npy"

samples = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_augmented_sound_samples.npy"

results = "/home/gamagee/workspace/gunshot_detection/REU_Data/testing_results/"

results_guns = results +"guns/"

results_others = results+"others/"


model = keras.models.load_model(model_path,custom_objects={'auc':auc})
model.summary()

sr = 22050

label_np = np.array(np.load(labels))
label_np = np.array(keras.utils.to_categorical(label_np, 2))
sample_np = np.array(np.load(samples)).reshape(-1, sampling_rate_per_two_seconds, 1)
a = np.argmax(model.predict(sample_np),axis=1)
b = np.argmax(label_np,axis=1)
diff = a-b

indexes = []
for i in range(len(diff)):
    if diff[i]!=0:
        indexes.append(i)
arr = np.array(np.load(samples))
arr_2 = np.array(np.load(labels))
for ind in indexes:
    if arr_2[ind]==1:
        direc = results_guns
    else:
        direc = results_others
    filepath = direc+"/"+str(ind)+".wav"
    print(filepath)
    librosa.output.write_wav(filepath,arr[ind],sr)
