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

# Dimension Reduction
from sklearn.decomposition import PCA

# Data Pre-processing
import pandas as pd
import soundfile
from sklearn.model_selection import KFold

# Deep Learning
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer

BASE_DIRECTORY = "/home/rjhosler/REU/Datasets/"

os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

samples = np.load(BASE_DIRECTORY + "gunshot_augmented_sound_samples.npy")
labels = np.load(BASE_DIRECTORY + "gunshot_augmented_sound_labels.npy")

sample_weights = np.array(
    [1 for normally_recorded_sample in range(len(samples) - 660)] + [15 for raspberry_pi_recorded_sample in range(660)])

labels = np.array([("gun_shot" if label == 1 else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels, 1 - labels))

'''
kf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in kf.split(samples):
    print("TRAIN:", train_index, "TEST:", test_index)
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]
    break

for test_index, valid_index in kf.split(test_wav):
    print("TEST:", test_index, "VALID:", valid_index)
    test_wav, valid_wav = test_wav[test_index], test_wav[valid_index]
    test_label, valid_label = test_label[test_index], test_label[valid_index]
    break
'''


all_index = np.arange(len(samples))
train_index = np.load("training_set_indexes.npy")
test_index = np.load("testing_set_indexes.npy")
valid_index = np.delete(all_index, list(train_index) + list(test_index))

print(train_index)
print(test_index)
print(valid_index)

train_wav, test_wav, valid_wav = samples[train_index], samples[test_index], samples[valid_index]
train_label, test_label, valid_label = labels[train_index], labels[test_index], labels[valid_index]
train_weights, test_weights, valid_weights = sample_weights[train_index], sample_weights[test_index], sample_weights[valid_index]

train_wav = train_wav.reshape(-1, 44100, 1)
test_wav = test_wav.reshape(-1, 44100, 1)
valid_wav = valid_wav.reshape(-1, 44100, 1)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

drop_out_rate = 0.25
learning_rate = 0.001
window = 8
small_window = 4
number_of_epochs = 100
number_of_classes = 2
batch_size = 32
optimizer = optimizers.Adam(learning_rate)
input_shape = (44100, 1)
input_tensor = Input(shape=input_shape)
metrics = [auc, "accuracy"]


# ## Model Architecture

# In[ ]:


x = layers.Conv1D(16, window, activation="relu", padding="same")(input_tensor)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.MaxPool1D(16)(x)
x = layers.Dropout(rate=drop_out_rate)(x)

x = layers.Conv1D(32, window, activation="relu", padding="same")(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv1D(32, small_window, activation="relu", padding="same")(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.MaxPool1D(8)(x)
x = layers.Dropout(rate=drop_out_rate)(x)

x = layers.Conv1D(64, window, activation="relu", padding="same")(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv1D(64, small_window, activation="relu", padding="same")(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.MaxPool1D(8)(x)
x = layers.Dropout(rate=drop_out_rate)(x)

'''
x = layers.Conv1D(128, window, activation="relu", padding="same")(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Conv1D(128, small_window, activation="relu", padding="same")(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.MaxPool1D(8)(x)
x = layers.Dropout(rate=drop_out_rate)(x)
'''

x = layers.Flatten()(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = Dropout(rate=drop_out_rate*2)(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = Dropout(rate=drop_out_rate*2)(x)

output_tensor = layers.Dense(number_of_classes, activation="softmax")(x)

model = tf.keras.Model(input_tensor, output_tensor)
model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=metrics)


# ## Configuring model properties

# In[ ]:


model_filename = BASE_DIRECTORY + "gunshot_1D_model.pkl"

model_callbacks = [
    EarlyStopping(monitor="val_acc",
                  patience=15,
                  verbose=1,
                  mode="max"),

    ModelCheckpoint(model_filename, monitor="val_acc",
                    verbose=1,
                    save_best_only=True,
                    mode="max"),

    ReduceLROnPlateau(monitor='val_loss', 
		      patience=6, 
                      factor=0.6, 
                      min_lr=1e-8)
]


# ### Optional debugging of the model's architecture

print(model.summary())

# ## Training & caching the model

History = model.fit(train_wav, train_label,
                    validation_data=[test_wav, test_label],
                    epochs=100,
                    callbacks=model_callbacks,
                    verbose=1,
                    batch_size=batch_size,
                    sample_weight=train_weights,
                    shuffle=True)

model.load_weights(model_filename)

model.save(BASE_DIRECTORY + "CurrentModels/RYAN_1D_model.h5")

#Testing model on validation data

y_test_pred = model.predict(valid_wav)
y_predicted_classes_test = y_test_pred.argmax(axis=-1)
y_actual_classes_test = valid_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)
print(100 - (len(wrong_examples[0]) / len(valid_wav)) * 100)
