import numpy as np
from scipy.fftpack import fft
from scipy import signal
import librosa
import os

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

from tensorflow.python.client import device_lib

print(os.getcwd())
cache_location = ""
sample_file = cache_location+"/gunshot_sound_samples.npy"
label_file = cache_location+"/gunshot_sound_labels.npy"
