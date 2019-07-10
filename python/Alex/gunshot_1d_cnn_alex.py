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
import soundfile
import re
import cv2
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer


# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K


# # Initialization of Variables

# In[ ]:


samples = []
labels = []
sound_file_names = []
sample_weights = []
sound_file_id = 0
gunshot_frequency_threshold = 0.25
sample_rate = 22050
sample_rate_per_two_seconds = 44100
base_dir = "/home/amorehe/Datasets/"
data_dir = base_dir + "REU_Samples_and_Labels/"
sound_data_dir = data_dir + "Samples/"


# # Data Pre-Processing

# ## Reading in the CSV file of descriptors for many kinds of sounds

# In[ ]:


sound_types = pd.read_csv(data_dir + "labels.csv")


# ## Reading in all of the sound data WAV files

# In[ ]:


print("...Parsing sound data...")

for file in os.listdir(sound_data_dir):
    if file.endswith(".wav"):
        try:
            # Adding 2 second-long samples to the list of samples
            sound_file_id = int(re.search(r'\d+', file).group())
            sample, sample_rate = librosa.load(sound_data_dir + file)
            sample_source = sound_types.loc[sound_types["ID"] == sound_file_id, "Source"].values[0]
            
            if len(sample) <= sample_rate_per_two_seconds:
                sample_weight = 1
                
                # Upscales the weights for samples recorded on the Raspberry Pi
                if "recorded_on_raspberry_pi" in sample_source:
                    sample_weight = 50

                sound_file_names.append(file)
                sample_weights.append(sample_weight)
                print("Added a sample weight of", sample_weight)
                
            else:
                for i in range(0, sample.size - sample_rate_per_two_seconds, sample_rate_per_two_seconds):
                    sample_weight = 1
                    
                    # Upscales the weights for samples recorded on the Raspberry Pi
                    if "recorded_on_raspberry_pi" in sample_source:
                        sample_weight = 50

                    sound_file_names.append(file)
                    sample_weights.append(sample_weight)
                    print("Added a sample weight of", sample_weight)

        except:
            sample, sample_rate = soundfile.read(sound_data_dir + file)
            print("Sound(s) not recognized by Librosa:", file)
            pass


# ## Caching NumPy arrays as NumPy files

# In[ ]:

np.save(base_dir + "gunshot_sound_file_names.npy", sound_file_names)
np.save(base_dir + "gunshot_sound_sample_weights.npy", sample_weights)


# ## Loading NumPy files as NumPy arrays

# In[ ]:


samples = np.load(base_dir + "gunshot_sound_samples.npy")
labels = np.load(base_dir + "gunshot_sound_labels.npy")


# ## Augmenting data (i.e. time shifting, speed changing, etc.)

# In[ ]:


samples = np.array(samples)
labels = np.array(labels)
number_of_augmentations = 5
augmented_samples = np.zeros((samples.shape[0] * (number_of_augmentations + 1), samples.shape[1]))
augmented_labels = np.zeros((labels.shape[0] * (number_of_augmentations + 1),))
augmented_sound_file_names = []
augmented_sample_weights = []
j = 0

for i in range (0, len(augmented_samples), (number_of_augmentations + 1)):
    file = sound_file_names[j]
    augmented_sample_weight = sample_weights[j]
    
    augmented_sound_file_names.append(file)
    augmented_sound_file_names.append(file)
    augmented_sound_file_names.append(file)
    augmented_sound_file_names.append(file)
    augmented_sound_file_names.append(file)
    augmented_sound_file_names.append(file)
    
    augmented_sample_weights.append(augmented_sample_weight)
    augmented_sample_weights.append(augmented_sample_weight)
    augmented_sample_weights.append(augmented_sample_weight)
    augmented_sample_weights.append(augmented_sample_weight)
    augmented_sample_weights.append(augmented_sample_weight)
    augmented_sample_weights.append(augmented_sample_weight)
    
    j += 1

    print("Added six sample weights of", augmented_sample_weight)

sound_file_names = np.array(augmented_sound_file_names)
sample_weights = np.array(augmented_sample_weights)


# ## Saving augmented NumPy arrays as NumPy files

# In[ ]:

np.save(base_dir + "gunshot_augmented_sound_file_names.npy", sound_file_names)
np.save(base_dir + "gunshot_augmented_sound_sample_weights.npy", sample_weights)


# ## Loading augmented NumPy files as NumPy arrays

# In[ ]:


samples = np.load(base_dir + "gunshot_augmented_sound_samples.npy")
labels = np.load(base_dir + "gunshot_augmented_sound_labels.npy")


# ## Restructuring the label data

# In[ ]:


labels = np.array([("gun_shot" if label == 1 else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels, 1 - labels))


# ### Optional debugging of the sample and label data's shape

# In[ ]:


print("Shape of samples array:", samples.shape)
print("Shape of labels array:", labels.shape)


# ## Arranging the data

# In[ ]:


kf = KFold(n_splits = 3, shuffle = True)
for train_index, test_index in kf.split(samples):
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]


# ## Reshaping the sound data

# In[ ]:


train_wav = train_wav.reshape(-1, sample_rate_per_two_seconds, 1)
test_wav = test_wav.reshape(-1, sample_rate_per_two_seconds, 1)


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
optimizer = optimizers.Adam(lr = 0.001, decay = 0.001 / 100)
input_tensor = Input(shape = (44100, 1))


# ## Configuration of GPU for training (optional)

# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
K.set_session(session)


# ## Model Architecture

# In[ ]:


x = layers.Conv1D(16, 9, activation = "relu", padding = "same")(input_tensor)
x = layers.Conv1D(16, 9, activation = "relu", padding = "same")(x)
x = layers.MaxPool1D(16)(x)
x = layers.Dropout(rate = 0.25)(x)

x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate = 0.25)(x)

x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate = 0.25)(x)

x = layers.Conv1D(256, 3, activation = "relu", padding = "same")(x)
x = layers.Conv1D(256, 3, activation = "relu", padding = "same")(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(rate = (0.5))(x) # Increasing drop-out rate here to prevent overfitting

x = layers.Dense(64, activation = "relu")(x)
x = layers.Dense(1028, activation = "relu")(x)
output_tensor = layers.Dense(2, activation = "softmax")(x)

model = tf.keras.Model(input_tensor, output_tensor)
model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = [auc, "accuracy"])


# ## Configuring model properties

# In[ ]:


model_filename = base_dir + "gunshot_sound_model.pkl"

model_callbacks = [
    EarlyStopping(monitor = 'val_acc',
                  patience = 15,
                  verbose = 1,
                  mode = 'max'),
    
    ModelCheckpoint(model_filename, monitor = 'val_acc',
                    verbose = 1,
                    save_best_only = True,
                    mode = 'max'),
]


# ### Optional debugging of the model's architecture

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
          sample_weight = sample_weights,
          shuffle = True)

model.save(base_dir + "gunshot_sound_model.h5")


# ### Optional debugging of incorrectly-labeled examples

# In[ ]:


y_test_pred = model.predict(test_wav)
y_predicted_classes_test = y_test_pred.argmax(axis = -1)
y_actual_classes_test = test_label.argmax(axis = -1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)


# ## Converting model to TensorFlow Lite format

# In[ ]:


model_name = base_dir + "gunshot_sound_model"
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model_name + ".h5")
converter.post_training_quantize = True
tflite_model = converter.convert()
open(model_name + ".tflite", "wb").write(tflite_model)
