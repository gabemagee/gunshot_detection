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
import librosa.display

# Dimension Reduction
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer

# Data Pre-processing
import pandas as pd
import soundfile
from sklearn.model_selection import KFold
import cv2
import matplotlib.pyplot as plt

# Deep Learning
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

samples = np.load("aug_samples.npy")

#Make spectrograms
def make_spectrogram(y,sr):
	return np.array(librosa.feature.melspectrogram(y=y, sr=sr))

sr = 22050
n = 0
for sample in samples:
    a = make_spectrogram(sample,sr)
    plt.interactive(False)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    librosa.display.specshow(librosa.power_to_db(a, ref=np.max))
    plt.savefig('./spectrograms/' + str(n) + '.png', dpi=400, bbox_inches='tight',pad_inches=0)
    n += 1

    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
