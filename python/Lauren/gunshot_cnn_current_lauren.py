
# In[3]:


# File Directory 
import glob
import os

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal

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
import librosa
import soundfile

# Deep Learning
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K



from keras.activations import relu, softmax
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras import losses, models, optimizers
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)





# # Data

# In[4]:


sampling_rate_per_two_seconds = 44100
input_shape = (sampling_rate_per_two_seconds, 1)


# In[5]:


#load the data from the files as numpy arrays
samples = np.load("/home/lauogden/data/gunshot_sound_samples.npy")
labels = np.load("/home/lauogden/data/gunshot_sound_labels.npy")


# In[ ]:


print("There are " + str(len(samples)) + " samples.")
print("There are " + str(len(labels)) + " labels.")


# # Split Data

# In[6]:


#TEST IF GPU IS AVILABLE?
print("TESTING IF GPU IS AVAILABLE:")
print(tf.test.gpu_device_name())


print(tf.Session())

#USING THE GPU:

with tf.device("/gpu:0"):
    #train test split

    kf = KFold(n_splits=3, shuffle=True)
    samples = np.array(samples)
    labels = np.array(labels)
    for train_index, test_index in kf.split(samples):
        train_wav, test_wav = samples[train_index], samples[test_index]
        train_label, test_label = labels[train_index], labels[test_index]


    # In[9]:


    #Reshape data
    train_wav = train_wav.reshape(-1,44100,1)
    test_wav = test_wav.reshape(-1,44100,1)
    train_label = keras.utils.to_categorical(train_label, 2)
    test_label = keras.utils.to_categorical(test_label, 2)


    # In[ ]:


    print(train_wav.shape)


    # # Model

    # In[10]:


    # Parameters
    lr = 0.001
    batch_size = 32
    drop_out_rate = 0.2


    # In[11]:


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
    output_tensor = layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(input_tensor, output_tensor)

    model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(lr = lr),
                 metrics=['accuracy'])


    # In[ ]:


    model_filename = '/home/lauogden/gunshot_cnn_model.pkl' 

    # add callbacks to stop early if it stops improving

    callbacks = [
        EarlyStopping(monitor='val_acc',
                      patience=10,
                      verbose=1,
                      mode='auto'),

        ModelCheckpoint(model_filename, monitor='val_acc',
                        verbose=1,
                        save_best_only=True,
                        mode='auto'),
    ]


    # In[15]:


    model.summary()


    # In[16]:


    #model.load("/home/lauogden/gunshot_cnn_model.h5")

    #FIT IT
    History = model.fit(train_wav, train_label, 
              validation_data=[test_wav, test_label],
              epochs=50,
              callbacks=model_callbacks,
              verbose=1,
              batch_size=batch_size,
              shuffle=True)

    model.save("/home/lauogden/data/gunshot_cnn_model.h5")
    print("\n \n saved model as gunshot_cnn_model.h5 \n \n")
    

    # In[26]:


    '''
    #incorrectly predicted data samples
    Y_test_pred = model.predict(test_wav)
    y_predicted_classes_test = y_test_pred.argmax(axis=-1)
    y_actual_classes_test= test_label.argmax(axis=-1)
    wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
    print("Wrong examples: ")
    print(wrong_examples)
    '''
    
    
    #attempt to convert to tflite?
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    print("created convertor")
    tflite_model = converter.convert()
    print("converted the model to tf lite")
    open("/home/lauogden/data/converted_model.tflite", "wb").write(tflite_model)
    print("saved the tf lite model")
    

