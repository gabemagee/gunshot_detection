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
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer


# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers, backend as K
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
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
base_dir = "/home/alexm/Datasets/"
data_dir = base_dir + "REU_Samples_and_Labels/"
spectrogram_dir = base_dir + "Spectrograms/"
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

print("The number of samples available for training is currently " + str(len(samples)) + '.')
print("The number of labels available for training is currently " + str(len(labels)) + '.')


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

    print("Added six samples weights of", augmented_sample_weight)

sound_file_names = np.array(augmented_sound_file_names)
sample_weights = np.array(augmented_sample_weights)


# ## Saving augmented NumPy arrays as NumPy files

# In[ ]:


np.save(base_dir + "gunshot_augmented_sound_file_names.npy", sound_file_names)
np.save(base_dir + "gunshot_augmented_sound_sample_weights.npy", sample_weights)

# ## Loading augmented NumPy file as a NumPy array

# In[ ]:


labels = np.load(base_dir + "gunshot_augmented_sound_labels.npy")


# ## Reading spectrograms into memory

# In[ ]:


spectrograms = []
for i in range(len(labels)):
    image = cv2.imread(spectrogram_dir + str(i) + ".png")
    image = cv2.resize(image, (192, 192))
    spectrograms.append(image)


# ## Restructuring spectrograms

# In[ ]:


samples = np.array(spectrograms).reshape(-1, 192, 192, 3)
samples = samples.astype("float32")
samples /= 255
print("Finished loading all spectrograms into memory...")


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


kf = KFold(n_splits = 3, shuffle = True)
for train_index, test_index in kf.split(samples):
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]


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
optimizer = Adam(lr = 0.001, decay = 0.001 / 100)
input_tensor = Input(shape = (192, 192))


# ## Configuration of GPU for training (optional)

# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
K.set_session(session)


# ## Model Architecture

# In[ ]:


""" Step 1: Instantiate a sequential model """

model = Sequential()


""" Step 2: Create the input and hidden layers """

# First Layer
model.add(Conv2D(32, (3, 3), padding = "same", input_shape = (192, 192, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Dropout(0.25))

# Second Layer: (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(64, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# Third Layer: (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(128, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

# Fourth Layer: (CONV => RELU) * 2 => POOL
model.add(Conv2D(256, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(Conv2D(256, (3, 3), padding = "same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (2, 2)))
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

model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = [auc, "accuracy"])


# ## Configuring model properties

# In[ ]:


model_filename = base_dir + "gunshot_2d_spectrogram_model.pkl"

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
          sample_weight = sample_weights,
          shuffle = True)

model.save(base_dir + "gunshot_2d_spectrogram_model.h5")


# ### Debugging of incorrectly-labeled examples (optional)

# In[ ]:


y_test_pred = model.predict(test_wav)
y_predicted_classes_test = y_test_pred.argmax(axis = -1)
y_actual_classes_test = test_label.argmax(axis = -1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)


# ## Converting model to TensorFlow Lite format

# In[ ]:


model_name = base_dir + "gunshot_2d_spectrogram_model"
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model_name + ".h5")
converter.post_training_quantize = True
tflite_model = converter.convert()
open(model_name + ".tflite", "wb").write(tflite_model)
