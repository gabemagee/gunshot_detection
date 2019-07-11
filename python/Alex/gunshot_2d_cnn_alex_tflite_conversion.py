#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# ### File Directory Libraries

# In[ ]:


import os

# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
from tensorflow.keras import Input, layers, backend as K
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam

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


# ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"
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

model = load_model(BASE_DIRECTORY + "gunshot_2d_spectrogram_model.h5", custom_objects={"auc": auc})

# ## Converting model to TensorFlow Lite format


# In[ ]:

model_name = BASE_DIRECTORY + "gunshot_2d_spectrogram_model"
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + ".h5", custom_objects={"auc": auc})
converter.post_training_quantize = True
tflite_model = converter.convert()
open(model_name + ".tflite", "wb").write(tflite_model)
