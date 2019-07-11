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


import librosa
import librosa.display
import cv2
import six
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
from tensorflow.keras import Input, layers, backend as K
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # Initialization of Variables

# In[ ]:


GUNSHOT_FREQUENCY_THESHOLD = 0.25
SAMPLE_RATE_PER_SECOND = 22050
SAMPLE_RATE_PER_TWO_SECONDS = 44100
SOUND_FILE_ID = 0
BASE_DIRECTORY = "/home/amorehe/Datasets/"
DATA_DIRECTORY = BASE_DIRECTORY + "REU_Samples_and_Labels/"
SPECTROGRAM_DIRECTORY = BASE_DIRECTORY + "Spectrograms/"
SOUND_DATA_DIRECTORY = DATA_DIRECTORY + "Samples/"
samples = []
labels = []
sound_file_names = []
sample_weights = []

# ## Loading augmented NumPy files as NumPy arrays

# In[ ]:


samples = np.load(BASE_DIRECTORY + "gunshot_augmented_sound_samples.npy")
labels = np.load(BASE_DIRECTORY + "gunshot_augmented_sound_labels.npy")

# ## Instantiating a sample weights NumPy array

# In[ ]:


sample_weights = np.array(
    [1 for normally_recorded_sample in range(len(samples) - 660)] + [50 for raspberry_pi_recorded_sample in range(660)])
print("Shape of samples weights before splitting:", sample_weights.shape)

# ## Defining spectrogram conversion functions

# In[ ]:


def convert_to_spectrogram(data, sample_rate):
    return np.array(librosa.feature.melspectrogram(y = data, sr = sample_rate), dtype = "float32")


def power_to_db(S, ref = 1.0, amin = 1e-10, top_db = 80.0):
    S = np.asarray(S)
    if amin <= 0:
        logger.debug('ParameterError: amin must be strictly positive')
    if np.issubdtype(S.dtype, np.complexfloating):
        logger.debug('Warning: power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(np.abs(D)**2) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S
    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)
    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    if top_db is not None:
        if top_db < 0:
            logger.debug('ParameterError: top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def convert_spectrogram_to_image(spectrogram):
    plt.interactive(False)
    
    figure = plt.figure(figsize = [0.72, 0.72], dpi = 400)
    plt.tight_layout(pad = 0)
    ax = figure.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    librosa.display.specshow(power_to_db(spectrogram, ref = np.max))
    
    canvas = FigureCanvas(figure)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    image = np.fromstring(figure.canvas.tostring_rgb(), dtype = "uint8")
    image = image.reshape((width, height, 3))
    image = cv2.resize(image, (192, 192))

    # Cleaning up the matplotlib instance
    plt.close()    
    figure.clf()
    plt.close(figure)
    plt.close("all")
    
    # Returns a NumPy array containing an image of a spectrogram
    return image

# ## Iteratively converting all augmented samples into spectrograms

# In[ ]:


spectograms = []

for sample in samples:
    spectrogram = convert_to_spectrogram(sample, SAMPLE_RATE_PER_SECOND)
    spectrogram = convert_spectrogram_to_image(spectrogram)
    spectrograms.append(spectrogram)
    print("Converted a sample into a spectrogram...")

# ## Restructuring spectrograms

# In[ ]:


samples = np.array(spectrograms).reshape(-1, 192, 192, 3)
samples = samples.astype("float32")
samples /= 255
print("Finished loading all spectrograms into memory...")

# ## Saving spectrograms as a NumPy array

# In[ ]:


np.save(BASE_DIRECTORY + "gunshot_augmented_sample_spectrograms.npy", samples)
print("Successfully saved all spectrograms as a NumPy array...")

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


kf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in kf.split(samples):
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]
    train_weights, test_weights = sample_weights[train_index], sample_weights[test_index]


# # Model


# ## ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"

# In[ ]:


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# ## Model Parameters

# In[ ]:


number_of_epochs = 100
batch_size = 32
optimizer = Adam(lr=0.001, decay=0.001 / 100)
input_tensor = Input(shape=(192, 192))

# ## Configuration of GPU for training (optional)

# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

# ## Model Architecture

# In[ ]:


""" Step 1: Instantiate a sequential model """

model = Sequential()

""" Step 2: Create the input and hidden layers """

# First Layer
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(192, 192, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

# Second Layer: (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Third Layer: (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fourth Layer: (CONV => RELU) * 2 => POOL
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

""" Step 3: Flatten the layers """

model.add(Flatten())

""" Step 4: Fully-connect the layers """

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Increasing dropout here to prevent overfitting

model.add(Dense(2))
model.add(Activation("softmax"))

""" Step 5: Compile the model """

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
                    mode="max")
]

# ### Debugging of the model's architecture (optional)

# In[ ]:


print(model.summary())

# ## Training & caching the model

# In[ ]:


History = model.fit(train_wav, train_label,
                    validation_data=[test_wav, test_label],
                    epochs=number_of_epochs,
                    callbacks=model_callbacks,
                    verbose=1,
                    batch_size=batch_size,
                    sample_weight=train_weights,
                    shuffle=True)

model.save(BASE_DIRECTORY + "gunshot_2d_spectrogram_model.h5")

# ### Debugging of incorrectly-labeled examples (optional)

# In[ ]:


y_test_pred = model.predict(test_wav)
y_predicted_classes_test = y_test_pred.argmax(axis=-1)
y_actual_classes_test = test_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)

# ## Converting model to TensorFlow Lite format

# In[ ]:


model_name = BASE_DIRECTORY + "gunshot_2d_spectrogram_model"
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + ".h5", custom_objects={"auc": auc})
converter.post_training_quantize = True
tflite_model = converter.convert()
open(model_name + ".tflite", "wb").write(tflite_model)
