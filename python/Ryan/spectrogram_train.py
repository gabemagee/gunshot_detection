#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# ### File Directory Libraries

import os

# ### Math Libraries

import numpy as np
import matplotlib.pyplot as plt

# ### Data Pre-Processing Libraries

import librosa
import librosa.display
import cv2
import six
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ### Deep Learning Libraries

import tensorflow as tf
from tensorflow.keras import Input, layers, backend as K
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Flatten, multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

GUNSHOT_FREQUENCY_THESHOLD = 0.25
SAMPLE_RATE_PER_SECOND = 22050
SAMPLE_RATE_PER_TWO_SECONDS = 44100
SOUND_FILE_ID = 0
BASE_DIRECTORY = "/home/rjhosler/REU/Datasets/"

# ## Loading augmented NumPy files as NumPy arrays

'''
samples = np.load(BASE_DIRECTORY + "gunshot_augmented_sample_spectrograms.npy")
labels = np.load(BASE_DIRECTORY + "gunshot_augmented_sound_labels.npy")
'''

train_wav = np.load(BASE_DIRECTORY + "128_128_augmented_training_spectrograms.npy")
test_wav = np.load(BASE_DIRECTORY + "128_128_augmented_testing_spectrograms.npy")
valid_wav = np.load(BASE_DIRECTORY + "128_128_augmented_validation_spectrograms.npy")

train_label = np.load(BASE_DIRECTORY + "augmented_training_labels.npy")
test_label = np.load(BASE_DIRECTORY + "augmented_testing_labels.npy")
valid_label = np.load(BASE_DIRECTORY + "augmented_validation_labels.npy")

print("Successfully loaded all spectrograms and labels as NumPy arrays...")

label_binarizer = LabelBinarizer()

train_label = np.array([("gun_shot" if label == "gun_shot" else "other") for label in train_label])
test_label = np.array([("gun_shot" if label == "gun_shot" else "other") for label in test_label])
valid_label = np.array([("gun_shot" if label == "gun_shot" else "other") for label in valid_label])

train_label = label_binarizer.fit_transform(train_label)
train_label = np.hstack((train_label, 1 - train_label))

test_label = label_binarizer.fit_transform(test_label)
test_label = np.hstack((test_label, 1 - test_label))

valid_label = label_binarizer.fit_transform(valid_label)
valid_label = np.hstack((valid_label, 1 - valid_label))

'''
sample_weights = np.array(
    [1 for normally_recorded_sample in range(len(samples) - 660)] + [15 for raspberry_pi_recorded_sample in range(660)])
'''

# # Model

# ## ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# ## Model Parameters

batch_size = 32
optimizer = Adam(lr=0.001)
input_tensor = Input(shape=(128, 128, 1))
window = (4, 4)

""" Step 2: Create the input and hidden layers """

# First Layer
x = Conv2D(16, window, padding="same", activation="relu")(input_tensor)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(4, 4))(x)
x = Dropout(0.25)(x)

# Second Layer: (CONV => RELU) * 2 => POOL
x = Conv2D(32, window, activation="relu", padding="same")(x)
x = BatchNormalization(axis=-1)(x)
x = Conv2D(32, window, activation="relu", padding="same")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

# Third Layer: (CONV => RELU) * 2 => POOL
x = Conv2D(64, window, activation="relu", padding="same")(x)
x = BatchNormalization(axis=-1)(x)
x = Conv2D(64, window, activation="relu", padding="same")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(4, 4))(x)
x = Dropout(0.25)(x)

'''
# Fourth Layer: (CONV => RELU) * 2 => POOL
x = Conv2D(256, window, activation="relu", padding="same")(x)
x = BatchNormalization(axis=-1)(x)
x = Conv2D(256, window, activation="relu", padding="same")(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
'''

""" Step 3: Flatten the layers """

x = Flatten()(x)

""" Step 4: Fully-connect the layers """

x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)  # Increasing dropout here to prevent overfitting

x = Dense(128, activation="relu")(x) 
x = BatchNormalization()(x)
x = Dropout(0.5)(x)  # Increasing dropout here to prevent overfitting

output_tensor = layers.Dense(2, activation="softmax")(x)

""" Step 5: Compile the model """
model = tf.keras.Model(input_tensor, output_tensor)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[auc, "accuracy"])
# ## Configuring model properties

# In[ ]:


model_filename = BASE_DIRECTORY + "gunshot_2d_spectrogram_model.pkl"

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


# In[ ]:


print(model.summary())

# ## Training & caching the model

# In[ ]:


History = model.fit(train_wav, train_label,
                    validation_data=[test_wav, test_label],
                    epochs=100,
                    callbacks=model_callbacks,
                    verbose=1,
                    batch_size=batch_size,
                    shuffle=True)

model.load_weights(model_filename)

model.save(BASE_DIRECTORY + "CurrentModels/128_128_RYAN_spectrogram_model.h5")


# ## Converting model to TensorFlow Lite format
'''
model_name = BASE_DIRECTORY + "gunshot_2d_spectrogram_model"
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + ".h5", custom_objects={"auc": auc})
converter.post_training_quantize = True
tflite_model = converter.convert()
open(model_name + ".tflite", "wb").write(tflite_model)
'''
# ### Debugging of incorrectly-labeled examples (optional)


model.load_weights(BASE_DIRECTORY + "CurrentModels/128_128_RYAN_spectrogram_model.h5")

y_test_pred = model.predict(valid_wav)
y_predicted_classes_test = y_test_pred.argmax(axis=-1)
y_actual_classes_test = valid_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)
print(100 - (len(wrong_examples[0]) / len(valid_wav)) * 100)
