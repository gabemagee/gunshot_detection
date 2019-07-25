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
samples = []
labels = []
sound_file_names = []
sample_weights = []

# ## Loading augmented NumPy files as NumPy arrays

# In[ ]:


samples = np.load(BASE_DIRECTORY + "gunshot_augmented_sample_spectrograms.npy")
labels = np.load(BASE_DIRECTORY + "gunshot_augmented_sound_labels.npy")

print("Successfully loaded all spectrograms and labels as NumPy arrays...")
print("Type of the spectrograms array:", samples.dtype)

# ## Instantiating a sample weights NumPy array

# In[ ]:


sample_weights = np.array(
    [1 for normally_recorded_sample in range(len(samples) - 660)] + [15 for raspberry_pi_recorded_sample in range(660)])
#print("Shape of samples weights before splitting:", sample_weights.shape)

# ## Restructuring the label data

# In[ ]:


labels = np.array([("gun_shot" if label == 1 else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels, 1 - labels))

# ### Debugging of the sample and label data's shape (optional)

# In[ ]:


#print("Shape of samples array:", samples.shape)
#print("Shape of labels array:", labels.shape)

# ## Arranging the data

# In[ ]:

'''
kf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in kf.split(samples):
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]
    train_weights, test_weights = sample_weights[train_index], sample_weights[test_index]
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


# # Model


# ## ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"

# In[ ]:


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# ## Model Parameters

# In[ ]:


number_of_epochs = 50
batch_size = 32
optimizer = Adam(lr=0.001)
input_tensor = Input(shape=(128, 64, 1))
window = (4, 4)

""" Step 2: Create the input and hidden layers """

# First Layer
x = Conv2D(16, window, padding="same", activation="relu")(input_tensor)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D(pool_size=(4, 2))(x)
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

'''
#Attention layer
attention_probs = Dense(1024, activation='softmax', name='attention_vec')(x)
attention_probs = Dropout(0.5)(attention_probs)
attention_mul = multiply([x, attention_probs], name='attention_mul')
attention_mul = Dense(256)(attention_mul)
'''

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
                    sample_weight=train_weights,
                    shuffle=True)

model.load_weights(model_filename)

model.save(BASE_DIRECTORY + "CurrentModels/RYAN_smaller_spectrogram_model.h5")


# ## Converting model to TensorFlow Lite format
'''
model_name = BASE_DIRECTORY + "gunshot_2d_spectrogram_model"
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + ".h5", custom_objects={"auc": auc})
converter.post_training_quantize = True
tflite_model = converter.convert()
open(model_name + ".tflite", "wb").write(tflite_model)
'''
# ### Debugging of incorrectly-labeled examples (optional)

# In[ ]:

model.load_weights(BASE_DIRECTORY + "CurrentModels/RYAN_smaller_spectrogram_model.h5")

y_test_pred = model.predict(valid_wav)
y_predicted_classes_test = y_test_pred.argmax(axis=-1)
y_actual_classes_test = valid_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)
print(100 - (len(wrong_examples[0]) / len(valid_wav)) * 100)
