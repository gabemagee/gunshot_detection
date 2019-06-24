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

import csv

os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def make_spectrogram(y,sr):
    return np.array(librosa.feature.melspectrogram(y=y, sr=sr))

number_of_desired_samples = 250
sampling_rate_per_two_seconds = 44100

model_path = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_sound_model_spectrograph_model.h5"

model_path_linear = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_sound_model.h5"

a_labels = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_augmented_sound_labels.npy"
a_samples = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_augmented_sound_samples.npy"


b_labels = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_augmented_sound_labels.npy"
b_samples = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_augmented_sound_samples.npy"


results = "/home/gamagee/workspace/gunshot_detection/REU_Data/testing_results/"

results_spectrogram = results+"spectro/"

results_regular = results+"regular/"

results_both = results+"both/"

spectrogram_samples_path = "/home/gamagee/workspace/gunshot_detection/REU_Data/spectro_samples.npy"
reg_samples_path = ""

model = keras.models.load_model(model_path,custom_objects={'auc':auc})
model.summary()

sr = 22050

label_np = np.concatenate((np.array(np.load(a_labels)),np.array(np.load(b_labels))))
scont = np.concatenate((np.array(np.load(a_samples)),np.array(np.load(a_samples))))


#spectro_samples = np.array([make_spectrogram(a,sr) for a in scont]).reshape(-1,128,87,1)

spectro_samples = np.array(np.load(spectrogram_samples_path)).reshape(-1, sampling_rate_per_two_seconds, 1)



print("done preprocessing")

label_np_1 = np.array(keras.utils.to_categorical(label_np, 2))
#sample_np = np.array(scont).reshape(-1, sampling_rate_per_two_seconds, 1)
predictions = model.predict(spectro_samples)
a = np.argmax(predictions,axis=1)
b = np.argmax(label_np_1,axis=1)
diff_s = a-b


#testing the other model
model = keras.models.load_model(model_path_linear,custom_objects={'auc':auc})
model.summary()
label_np_1 = np.array(keras.utils.to_categorical(label_np, 2))
sample_np = np.array(scont).reshape(-1, sampling_rate_per_two_seconds, 1)
predictions = model.predict(sample_np)
a = np.argmax(predictions,axis=1)
b = np.argmax(label_np_1,axis=1)
diff_a = a - b


indexes_both = []
indexes_s = []
indexes_a = []
for i in range(len(diff_s)):
    if diff_s[i]!=0 and diff_a[i]!=0:
        indexes_both.append(i)
    if diff_s[i]!=0:
        indexes_s.append(i)
    if diff_a[i]!=0:
        indexes_a.append(i)
for ind in indexes_both:
    if label_np[ind]==1:
        direc = results_both+"false_negative/"
    else:
        direc = results_both+"false_positive/"
    filepath = direc+"/"+str(ind)+".wav"
    print(filepath)
    librosa.output.write_wav(filepath,scont[ind],sr)
for ind in indexes_s:
    if label_np[ind]==1:
        direc = results_spectrogram+"false_negative/"
    else:
        direc = results_spectrogram+"false_positive/"
    filepath = direc+"/"+str(ind)+".wav"
    print(filepath)
    librosa.output.write_wav(filepath,scont[ind],sr)
for ind in indexes_a:
    if label_np[ind]==1:
        direc = results_regular+"false_negative/"
    else:
        direc = results_regular+"false_positive/"
    filepath = direc+"/"+str(ind)+".wav"
    print(filepath)
    librosa.output.write_wav(filepath,scont[ind],sr)
