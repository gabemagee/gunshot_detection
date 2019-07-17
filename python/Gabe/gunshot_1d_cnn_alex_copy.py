#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# ### File Directory Libraries

# In[ ]:


import os

# ### Math Libraries

# In[ ]:


import numpy as np

# ### Data Pre-Processing Libraries

# In[ ]:


from array import array
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer

# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # Initialization of Variables

# In[ ]:


SAMPLE_RATE_PER_SECOND = 22050
SAMPLE_RATE_PER_TWO_SECONDS = 44100
BASE_DIRECTORY = "/home/amorehe/Datasets/"

# ## Loading augmented NumPy files as NumPy arrays

# In[ ]:
base_dir = "/home/gamagee/workspace/gunshot_detection/"
model_dir = base_dir+"raspberry_pi/models/"

sample_dir = base_dir+"REU_Data/spectrogram_training/samples_and_labels/"

label_path = sample_dir+"gunshot_augmented_sound_labels.npy"

sample_path = sample_dir+"gunshot_augmented_sound_samples.npy"



samples = np.load(sample_path)
labels = np.load(label_path)
#sound_file_names = np.load(BASE_DIRECTORY + "gunshot_augmented_sound_file_names.npy")

# ## Instantiating a sample weights NumPy array

# In[ ]:


sample_weights = np.array(
    [1 for normally_recorded_sample in range(len(samples) - 660)] + [20 for raspberry_pi_recorded_sample in range(660)])
print("Shape of samples weights before splitting:", sample_weights.shape)

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

testing_indexes_path = base_dir+"raspberry_pi/indexes/testing_set_indexes.npy"

testing_indexes = np.load(testing_indexes_path)

training_indexes_path = base_dir+"raspberry_pi/indexes/training_set_indexes.npy"

training_indexes = np.load(training_indexes_path)

print("~~~~~~~~~~~~~~~~")

train_wav = []
train_label = []
train_weights = []
test_wav = []
test_label = []
test_weights = []
validation_wav = []
validation_label = []
validation_weights = []

for i in range(len(labels)):
    if i in training_indexes:
        train_wav.append(samples[i])
        train_label.append(labels[i])
        train_weights.append(sample_weights[i])
    elif i in testing_indexes:
        test_wav.append(samples[i])
        test_label.append(labels[i])
        test_weights.append(sample_weights[i])
    else:
        validation_wav.append(samples[i])
        validation_label.append(labels[i])
        validation_weights.append(sample_weights[i])

train_wav = np.array(train_wav)
train_label = np.array(train_label)
train_weights = np.array(train_weights)
test_wav = np.array(test_wav)
test_label = np.array(test_label)
test_weights = np.array(test_weights)
validation_wav = np.array(validation_wav)
validation_label = np.array(validation_label)
validation_weights = np.array(validation_weights)

print("finished split")

"""
kf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in kf.split(samples):
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]
    train_weights, test_weights = sample_weights[train_index], sample_weights[test_index]
"""
# ## Reshaping the sound data

# In[ ]:


train_wav = train_wav.reshape(-1, SAMPLE_RATE_PER_TWO_SECONDS, 1)
test_wav = test_wav.reshape(-1, SAMPLE_RATE_PER_TWO_SECONDS, 1)


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
optimizer = optimizers.Adam(lr=0.001, decay=0.001 / 100)
input_tensor = Input(shape=(44100, 1))

# ## Configuration of GPU for training (optional)

# In[ ]:


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# K.set_session(session)

# ## Model Architecture

# In[ ]:


x = layers.Conv1D(16, 9, activation="relu", padding="same")(input_tensor)
x = layers.Conv1D(16, 9, activation="relu", padding="same")(x)
x = layers.MaxPool1D(16)(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(rate=(0.5))(x)  # Increasing drop-out rate here to prevent overfitting

x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(1028, activation="relu")(x)
output_tensor = layers.Dense(2, activation="softmax")(x)

model = tf.keras.Model(input_tensor, output_tensor)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[auc, "accuracy"])

# ## Configuring model properties

# In[ ]:


model_filename = BASE_DIRECTORY + "gunshot_sound_model.pkl"

model_callbacks = [
    EarlyStopping(monitor='val_acc',
                  patience=15,
                  verbose=1,
                  mode='max'),

    ModelCheckpoint(model_filename, monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    mode='max'),
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



model.save(BASE_DIRECTORY + "gunshot_sound_model_1d.h5")

print("saved at: ",BASE_DIRECTORY + "gunshot_sound_model.h5")

# ### Debugging of incorrectly-labeled examples (optional)

# In[ ]:


y_test_pred = model.predict(test_wav)
y_predicted_classes_test = y_test_pred.argmax(axis=-1)
y_actual_classes_test = test_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)

# ## Converting model to TensorFlow Lite format

# In[ ]:


model_name = BASE_DIRECTORY + "gunshot_sound_model"
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + ".h5", custom_objects={"auc": auc})
converter.post_training_quantize = True
tflite_model = converter.convert()
open(model_name + ".tflite", "wb").write(tflite_model)
