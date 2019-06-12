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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

from tensorflow.python.client import device_lib

sampling_rate_per_two_seconds = 44100
input_shape = (sampling_rate_per_two_seconds, 1)
cache_location = "/home/gamagee/workspace/gunshot_detection/REU_Data/Cache"
sample_file = cache_location+"/gunshot_sound_samples.npy"
label_file = cache_location+"/gunshot_sound_labels.npy"

samples = np.load(sample_file)
labels = np.load(label_file)

i = 0  # You can change the value of 'i' to adjust which sample is being inspected.
sample=samples[i]
sample_rate=22050
print("The number of samples available to the model for training is " + str(len(samples)) + '.')
print("The maximum frequency value in sample slice #" + str(i) + " is " + str(np.max(abs(sample))) + '.')
print("The label associated with sample slice #" + str(i) + " is " + str(labels[i]) + '.')
ipd.Audio(sample, rate=sample_rate)


# ## Arranging the data

# In[7]:





# ## Model Architecture

# In[11]:

with tf.device("/gpu:0"):
    kf = KFold(n_splits=3, shuffle=True)
    samples = np.array(samples)
    labels = np.array(labels)
    for train_index, test_index in kf.split(samples):
        train_wav, test_wav = samples[train_index], samples[test_index]
        train_label, test_label = labels[train_index], labels[test_index]


    # ## Reshaping/restructuring the data

    # In[9]:


    train_wav = train_wav.reshape(-1, sampling_rate_per_two_seconds, 1)
    test_wav = test_wav.reshape(-1, sampling_rate_per_two_seconds, 1)
    train_label = keras.utils.to_categorical(train_label, 2)
    test_label = keras.utils.to_categorical(test_label, 2)


    # ### Optional debugging of the training data's shape

    # In[10]:


    print(train_wav.shape)


    # # Model

    # ## Model Parameters

    # In[8]:


    learning_rate = 0.001
    batch_size = 32
    drop_out_rate = 0.2

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

    model.compile(loss=keras.losses.binary_crossentropy,
                 optimizer=keras.optimizers.Adam(lr = learning_rate),
                 metrics=['accuracy'])


    # ## Configuring model properties

    # In[12]:


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




    # ### Optional debugging of the model's architecture

    # In[ ]:


    model.summary()


    # ## Training & caching the model

    # In[13]:




    print(train_wav.shape,train_label.shape,test_wav.shape,test_label.shape)

    History = model.fit(train_wav, train_label,
              validation_data=[test_wav, test_label],
              epochs=50,
              callbacks=model_callbacks,
              verbose=1,
              batch_size=batch_size,
              shuffle=True)

    model.save("/home/gamagee/workspace/gunshot_detection/REU_Data/gunshot_sound_full_model.h5")


    # ## Summarizing history for accuracy

    # In[ ]:


    plt.plot(History.history['acc'])
    plt.plot(History.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


    # ## Summarizing history for loss

    # In[ ]:


    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


    # ### Optional debugging of incorrectly-labeled examples

    # In[ ]:


    y_test_pred = model.predict(test_wav)
    y_predicted_classes_test = y_test_pred.argmax(axis=-1)
    y_actual_classes_test= test_label.argmax(axis=-1)
    wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
    print(wrong_examples)


    # ### Optional debugging of an individual incorrectly-labeled example

    # In[ ]:


    i = 323
    sample = np.reshape(test_wav[i], sampling_rate_per_two_seconds, )
    sample_rate = 22050
    print(y_actual_classes_test[i], y_predicted_classes_test[i])
    ipd.Audio(sample, rate=sample_rate)
