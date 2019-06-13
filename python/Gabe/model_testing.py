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

print(os.getcwd())

model_path = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_sound_model.h5"

labels = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_augmented_sound_labels.npy"

samples = "/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_augmented_sound_samples.npy"

model = keras.models.load_model(model_path)
model.summary()
#loss, acc = new_model.evaluate(test_images, test_labels)
#print("Restored model, accuracy: {:5.2f}%".format(100*acc))

label_np = np.open(labels)
