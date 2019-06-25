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

#samples = np.load("aug_samples.npy")
labels = np.load("aug_labels.npy")
nclass = 13

#read in spectrograms
sa = []
for i in range(len(labels)):
    spec_dir = './spectrograms/'
    im = cv2.imread(spec_dir + str(i) + '.png')
    sa.append(im)

samples = np.array(sa).reshape(-1, 221, 223, 3)
samples = samples.astype('float32')
samples /= 255

'''
#4 categories
urban = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'other',
       'drilling', 'engine_idling', 'jackhammer', 'siren', 'street_music']
for i in range(len(labels)):
    if labels[i] in urban:
        labels[i] = 'other'
nclass = 4
'''

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#split data
kf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in kf.split(samples):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]
    break

# Parameters
lr = 0.001
generations = 20000
num_gens_to_wait = 250
batch_size = 32
drop_out_rate = 0.2
input_shape = (221, 223)

#For Conv2D add Channel
#train_wav = train_wav.reshape(-1, input_shape[0], input_shape[1], 3)
#test_wav = test_wav.reshape(-1, input_shape[0], input_shape[1], 3)
#train_wav = train_wav.astype('float32')
#test_wav = test_wav.astype('float32')

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# Initialising the CNN
n_labels = nclass
input_size = input_shape
model = Sequential()
chanDim = -1
channel = 3

# Layer 1
model.add(Conv2D(32, (3, 3), padding = "same", input_shape = (input_size[0], input_size[1], channel)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = chanDim))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = chanDim))
model.add(Conv2D(64, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = chanDim))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = chanDim))
model.add(Conv2D(128, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = chanDim))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(200, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = chanDim))
model.add(Conv2D(200, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = chanDim))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
    
# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(n_labels))
model.add(Activation("softmax"))

# Compiling the CNN
EPOCHS = 100
INIT_LR = 1e-3
opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ['accuracy'])

model_filename = '2Dspec.pkl' 
callbacks = [
    EarlyStopping(monitor='val_acc',
                  patience=15,
                  verbose=1,
                  mode='max'),
    
    ModelCheckpoint(model_filename, monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    mode='max'),
]

print(model.summary())

model.fit(train_wav, train_label, 
          validation_data=[test_wav, test_label],
          batch_size=batch_size,
          callbacks=callbacks,
          epochs=100,
          verbose=1)

model.load_weights(model_filename)

model.save("model.h5")