#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# ### File Directory Libraries

# In[ ]:


import os


# ### Math Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# ### Data Pre-Processing Libraries

# In[ ]:


import pandas as pd
import librosa
import librosa.display
import soundfile
import re
import cv2
import six
from array import array
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# ### Visualization Libraries

# In[ ]:


import IPython.display as ipd


# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
import kutilities.layers as kutil_layers
from tensorflow.keras import Input, layers, backend as K
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Dropout, Activation, BatchNormalization, Flatten, Lambda, Concatenate
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# # Initialization of Variables

# In[ ]:


GUNSHOT_FREQUENCY_THESHOLD = 0.25
SAMPLE_RATE_PER_SECOND = 22050
SAMPLE_RATE_PER_TWO_SECONDS = 44100
HOP_LENGTH = 345 * 2
MINIMUM_FREQUENCY = 20
MAXIMUM_FREQUENCY = SAMPLE_RATE_PER_SECOND
NUMBER_OF_MELS = 128
NUMBER_OF_FFTS = NUMBER_OF_MELS * 20
BASE_DIRECTORY = "/home/amorehe/Datasets/"
DATA_DIRECTORY = BASE_DIRECTORY + "REU_Samples_and_Labels/"
SPECTROGRAM_DIRECTORY = BASE_DIRECTORY + "Spectrograms/"
SOUND_DATA_DIRECTORY = DATA_DIRECTORY + "Samples/"
samples = []
labels = []
sound_file_names = []


# ## Loading a NumPy file as spectrograms

# In[ ]:


samples = np.load(BASE_DIRECTORY + "gunshot_augmented_sample_spectrograms.npy")
labels = np.load(BASE_DIRECTORY + "gunshot_augmented_sound_labels.npy")
print("Successfully loaded all spectrograms as a NumPy array...")

# ## Instantiating a sample weights NumPy array

# In[ ]:


sample_weights = np.array([1 for normally_recorded_sample in range(len(samples) - 660)] + [20 for raspberry_pi_recorded_sample in range(660)])
print("Shape of samples weights before splitting:", sample_weights.shape)


# ## Establishing index values for the data

# In[ ]:


all_index = np.arange(len(samples))
train_index = np.load("../../../raspberry_pi/indexes/training_set_indexes.npy")
test_index = np.load("../../../raspberry_pi/indexes/testing_set_indexes.npy")
valid_index = np.delete(all_index, list(train_index) + list(test_index))


# ## Restructuring the label data

# In[ ]:


labels = np.array([("gun_shot" if label == 1 else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels, 1 - labels))


# ### Debugging of the sample and label data's shape (optional)

# In[ ]:


print("Shape of samples array:", samples.shape)
print("Shape of labels array:", labels.shape)


# ## Arranging the data

# In[ ]:


train_wav, test_wav, valid_wav = samples[train_index], samples[test_index], samples[valid_index]
train_label, test_label, valid_label = labels[train_index], labels[test_index], labels[valid_index]
train_weights, test_weights, valid_weights = sample_weights[train_index], sample_weights[test_index], sample_weights[valid_index]


# # Model

# ## ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"

# In[ ]:


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# In[ ]:


def crosschannelnormalization(alpha = 1e-4, k = 2, beta = 0.75, n = 5,**kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
                                              , (0,half))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)


# In[ ]:


def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


# In[ ]:


def attention_control(args):
    x, dense_2 = args
    find_att = K.reshape(x, (15,15,10))
    find_att = K.transpose(find_att[:,:,:])
    find_att = K.mean(find_att, axis = 0)
    find_att = find_att / K.sum(find_att, axis = 0)
    find_att = K.repeat_elements(find_att, 32, axis = 0)
    find_att = K.reshape(find_att, (1, 32, 15, 15))
    return find_att

def no_attention_control(args):
    x, dense_2 = args
    find_att = K.ones(shape = (1, 32, 15, 15))
    return find_att


# In[ ]:


def change_shape1(x):
    x = K.reshape(K.transpose(x), (15 * 15, 32))
    return x

def att_shape(input_shape):
    return (input_shape[0][0], 32, 15, 15)

def att_shape2(input_shape):
    return input_shape[0][0:4]


# ## Model Parameters

# In[ ]:


number_of_epochs = 100
batch_size = 32
optimizer = Adam(lr = 0.001, decay = 0.001 / 100)


# ## Configuration of GPU for training (optional)

# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
K.set_session(session)


# ## Model Architecture

# In[ ]:


def attention_cnn(attention = True, image_size = "128, 64, 1"):
    # Make layers
    inputs = Input(shape = (1, image_size, image_size), name='input')

    conv_1a = Convolution2D(32, 3, 3, activation='relu', name='conv_1')
    maxp_1a = MaxPooling2D((3, 3), strides = (2,2), name = 'convmax_1')
    norm_1a = crosschannelnormalization(name = "convpool_1")
    zero_1a = ZeroPadding2D((2,2), name = 'convzero_1')

    conv_2a = Convolution2D(32, 3, 3, activation = 'relu', name = 'conv_2')
    maxp_2a = MaxPooling2D((3, 3), strides = (2,2), name = 'convmax_2')
    norm_2a = crosschannelnormalization(name = "convpool_2")
    zero_2a = ZeroPadding2D((2,2), name = 'convzero_2')

    dense_1a = Lambda(global_average_pooling, output_shape = global_average_pooling_shape, name='dense_1')
    dense_2a = Dense(10, activation = 'softmax', init = 'uniform', name = 'dense_2')

    # Make actual model
    input_pad = ZeroPadding2D((1,1), input_shape = (1, image_size, image_size), name = 'input_pad')(inputs)

    conv_1 = conv_1a(input_pad)
    conv_1 = maxp_1a(conv_1)
    conv_1 = norm_1a(conv_1)
    conv_1 = zero_1a(conv_1)

    conv_2_x = conv_2a(conv_1)
    conv_2 = maxp_2a(conv_2_x)
    conv_2 = norm_2a(conv_2)
    conv_2 = zero_2a(conv_2)
    conv_2 = Dropout(0.5)(conv_2)

    dense_1 = dense_1a(conv_2)
    dense_2 = dense_2a(dense_1)

    conv_shape1 = Lambda(change_shape1,output_shape = (32,), name = 'chg_shape')(conv_2_x)
    find_att = dense_2a(conv_shape1)

    if attention:
        find_att = Lambda(attention_control, output_shape = att_shape,name = 'att_con')([find_att,dense_2])
    else:
        find_att = Lambda(no_attention_control, output_shape = att_shape, name = 'att_con')([find_att,dense_2])

    zero_3a = ZeroPadding2D((1,1), name = 'convzero_3')(find_att)
    apply_attention = Concatenate(mode = 'mul', name = 'attend')([zero_3a,conv_1])

    conv_3 = conv_2a(apply_attention)
    conv_3 = maxp_2a(conv_3)
    conv_3 = norm_2a(conv_3)
    conv_3 = zero_2a(conv_3)

    dense_3 = dense_1a(conv_3)
    dense_4 = dense_2a(dense_3)

    model = Model(input = inputs, output = dense_4)

    return model

model = attention_cnn()
model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = [auc, "accuracy"])


# ## Configuring model properties

# In[ ]:


model_filename = BASE_DIRECTORY + "2D_attention_cnn.pkl"

model_callbacks = [
    EarlyStopping(monitor = "val_acc",
                  patience = 15,
                  verbose = 1,
                  mode = "max"),
    
    ModelCheckpoint(model_filename, monitor = "val_acc",
                    verbose = 1,
                    save_best_only = True,
                    mode = "max")
]


# ### Debugging of the model's architecture (optional)

# In[ ]:


print(model.summary())


# ## Training & caching the model

# In[ ]:


History = model.fit(train_wav, train_label, 
          validation_data = [test_wav, test_label],
          epochs = number_of_epochs,
          callbacks = model_callbacks,
          verbose = 1,
          batch_size = batch_size,
          sample_weight = train_weights,
          shuffle = True)

model.save(BASE_DIRECTORY + "2D_attention_cnn.h5")


# ### Debugging of incorrectly-labeled examples (optional)

# In[ ]:


y_val_pred = model.predict(valid_wav)
y_predicted_classes_val = y_val_pred.argmax(axis=-1)
y_actual_classes_val = valid_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_val != y_actual_classes_val)
print("Validation samples labeled incorrectly:", wrong_examples)
print("Validation accuracy of the current model:", 100 - (len(wrong_examples[0]) / len(valid_wav)) * 100)


# ## Converting model to TensorFlow Lite format

# In[ ]:


model_name = BASE_DIRECTORY + "gunshot_2d_spectrogram_model"
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + ".h5", custom_objects = {"auc" : auc})
tflite_model = converter.convert()
open(model_name + ".tflite", "wb").write(tflite_model)

