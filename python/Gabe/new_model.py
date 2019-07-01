#Library Imports
#File Directory Libraries
import os
import os
#Math Libraries
import numpy as np
import matplotlib.pyplot as plt
#Data Pre-Processing Libraries
import pandas as pd
import librosa
import soundfile
import re
import cv2
from sklearn.model_selection import KFold
#Visualization Libraries
import IPython.display as ipd
#Deep Learning Libraries
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#Configuration of Imported Libraries
#%matplotlib inline
#Initialization of Variables
samples=[]
labels = []
gunshot_frequency_threshold = 0.25
sample_rate = 22050
sample_rate_per_two_seconds = 44100
input_shape = (sample_rate_per_two_seconds, 1)
base_dir = "/home/gamagee/workspace/gunshot_detection/"
data_dir = base_dir + "REU_Samples_and_Labels/"
sound_data_dir = data_dir + "Samples/"
#Data Pre-Processing
#Reading in the CSV file of descriptors for many kinds of sounds
sound_types = pd.read_csv(data_dir + "labels.csv")
#Reading in all of the sound data WAV files
print("...Parsing sound data...")
sound_file_id = 0
sound_file_names = []
​
for file in os.listdir(sound_data_dir):
    if file.endswith(".wav"):
        try:
            # Adding 2 second-long samples to the list of samples
            sound_file_id = int(re.search(r'\d+', file).group())
            sample, sample_rate = librosa.load(sound_data_dir + file)
            prescribed_label = sound_types.loc[sound_types["ID"] == sound_file_id, "Class"].values[0]

            if len(sample) <= sample_rate_per_two_seconds:
                label = 1
                number_of_missing_hertz = sample_rate_per_two_seconds - len(sample)
                padded_sample = np.array(sample.tolist() + [0 for i in range(number_of_missing_hertz)])
                if prescribed_label != "gun_shot":
                    label = 0
​
                samples.append(padded_sample)
                labels.append(label)
                sound_file_names.append(file)
            else:
                for i in range(0, sample.size - sample_rate_per_two_seconds, sample_rate_per_two_seconds):
                    sample_slice = sample[i : i + sample_rate_per_two_seconds]
                    if prescribed_label != "gun_shot":
                        label = 0
                    elif np.max(abs(sample_slice)) < gunshot_frequency_threshold:
                        label = 0
​
                    samples.append(sample_slice)
                    labels.append(label)
                    sound_file_names.append(file)
​
        except:
            sample, sample_rate = soundfile.read(sound_data_dir + file)
            print("Sound(s) not recognized by Librosa:", file)
            pass
​
print("The number of samples available for training is currently " + str(len(samples)) + '.')
print("The number of labels available for training is currently " + str(len(labels)) + '.')
#Saving samples and labels as numpy array files
np.save(base_dir + "gunshot_sound_samples.npy", samples)
np.save(base_dir + "gunshot_sound_labels.npy", labels)
#Loading sample file and label file as numpy arrays

exit()

samples = np.load(base_dir + "gunshot_sound_samples.npy")
labels = np.load(base_dir + "gunshot_sound_labels.npy")
#Data augmentation functions
def time_shift(wav):
    start_ = int(np.random.uniform(-wav.shape[0] * 0.5, wav.shape[0] * 0.5))
    if start_ >= 0:
        wav_time_shift = np.r_[wav[start_:], np.random.uniform(-0.001, 0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001, 0.001, -start_), wav[:start_]]
    return wav_time_shift

def change_pitch(wav, sample_rate):
    magnitude = int(np.random.uniform(-10, 10))
    wav_pitch_change = librosa.effects.pitch_shift(wav, sample_rate, magnitude)
    return wav_pitch_change

def speed_change(wav):
    speed_rate = np.random.uniform(0.7, 1.3)
    wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()

    if len(wav_speed_tune) < len(wav):
        pad_len = len(wav) - len(wav_speed_tune)
        wav_speed_tune = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len / 2)),
                               wav_speed_tune,
                               np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2)))]
    else:
        cut_len = len(wav_speed_tune) - len(wav)
        wav_speed_tune = wav_speed_tune[int(cut_len / 2) : int(cut_len / 2) + len(wav)]
    return wav_speed_tune

def change_volume(wav, magnitude):
    # 0 < x < 1 quieter; x = 1 identity; x > 1 louder
    wav_volume_change = np.multiply(np.array([magnitude]), wav)
    return wav_volume_change

def add_background(wav, file, data_directory, label_to_avoid):
    label_csv = data_directory + "labels.csv"
    sound_directory = data_directory + "Samples/"
    sound_types = pd.read_csv(label_csv)
    bg_files = os.listdir(sound_directory)
    bg_files.remove(file)
    chosen_bg_file = bg_files[np.random.randint(len(bg_files))]
    jndex = int(chosen_bg_file.split('.')[0])
    while sound_types.loc[sound_types["ID"] == jndex, "Class"].values[0] == label_to_avoid:
        chosen_bg_file = bg_files[np.random.randint(len(bg_files))]
        jndex = int(chosen_bg_file.split('.')[0])
    bg, sr = librosa.load(sound_directory + chosen_bg_file)
    ceil = max((bg.shape[0] - wav.shape[0]), 1)
    start_ = np.random.randint(ceil)
    bg_slice = bg[start_ : start_ + wav.shape[0]]
    if bg_slice.shape[0] < wav.shape[0]:
        pad_len = wav.shape[0] - bg_slice.shape[0]
        bg_slice = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len / 2)), bg_slice, np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2)))]
    wav_with_bg = wav * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.5)
    return wav_with_bg
#Augmenting data (i.e. time shifting, speed changing, etc.)
samples = np.array(samples)
labels = np.array(labels)
number_of_augmentations = 5
augmented_samples = np.zeros((samples.shape[0] * (number_of_augmentations + 1), samples.shape[1]))
augmented_labels = np.zeros((labels.shape[0] * (number_of_augmentations + 1),))
j = 0
​
for i in range (0, len(augmented_samples), (number_of_augmentations + 1)):
    file = sound_file_names[j]

    augmented_samples[i,:] = samples[j,:]
    augmented_samples[i + 1,:] = time_shift(samples[j,:])
    augmented_samples[i + 2,:] = change_pitch(samples[j,:], sample_rate)
    augmented_samples[i + 3,:] = speed_change(samples[j,:])
    augmented_samples[i + 4,:] = change_volume(samples[j,:], np.random.uniform())
    if labels[j] == 1:
        augmented_samples[i + 5,:] = add_background(samples[j,:], file, sound_data_dir, "")
    else:
        augmented_samples[i + 5,:] = add_background(samples[j,:], file, sound_data_dir, "gun_shot")

    augmented_labels[i] = labels[j]
    augmented_labels[i + 1] = labels[j]
    augmented_labels[i + 2] = labels[j]
    augmented_labels[i + 3] = labels[j]
    augmented_labels[i + 4] = labels[j]
    augmented_labels[i + 5] = labels[j]
    j += 1
​
samples = augmented_samples
labels = augmented_labels
​
print("The number of samples available for training is currently " + str(len(samples)) + '.')
print("The number of labels available for training is currently " + str(len(labels)) + '.')
#Saving augmented samples and labels as numpy array files
np.save(base_dir + "gunshot_augmented_sound_samples.npy", samples)
np.save(base_dir + "gunshot_augmented_sound_labels.npy", labels)
#Loading augmented sample file and label file as numpy arrays

exit()

samples = np.load(base_dir + "gunshot_augmented_sound_samples.npy")
labels = np.load(base_dir + "gunshot_augmented_sound_labels.npy")
#Optional debugging after processing the data
i = 0  # You can change the value of 'i' to adjust which sample is being inspected.
sample=samples[i]
print("The number of samples available to the model for training is " + str(len(samples)) + '.')
print("The maximum frequency value in sample slice #" + str(i) + " is " + str(np.max(abs(sample))) + '.')
print("The label associated with sample slice #" + str(i) + " is " + str(labels[i]) + '.')
ipd.Audio(sample, rate=sample_rate)
#Restructuring the label data
labels = keras.utils.to_categorical(labels, 2)
#Optional debugging of the label data's shape
print(labels.shape)
#Arranging the data
kf = KFold(n_splits=3, shuffle=True)
samples = np.array(samples)
labels = np.array(labels)
for train_index, test_index in kf.split(samples):
    train_wav, test_wav = samples[train_index], samples[test_index]
    train_label, test_label = labels[train_index], labels[test_index]
#Reshaping the sound data
train_wav = train_wav.reshape(-1, sample_rate_per_two_seconds, 1)
test_wav = test_wav.reshape(-1, sample_rate_per_two_seconds, 1)
#Optional debugging of the sound data's shape
print(train_wav.shape)
#Model
#Loading previous model
model = load_model(base_dir + "gunshot_sound_model.h5")
#ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
#Model Parameters
drop_out_rate = 0.1
learning_rate = 0.001
number_of_epochs = 100
number_of_classes = 2
batch_size = 32
optimizer = optimizers.Adam(learning_rate, learning_rate / 100)
input_tensor = Input(shape=input_shape)
metrics = [auc, "accuracy"]
#Model Architecture
x = layers.Conv1D(16, 9, activation="relu", padding="same")(input_tensor)
x = layers.Conv1D(16, 9, activation="relu", padding="same")(x)
x = layers.MaxPool1D(16)(x)
x = layers.Dropout(rate=drop_out_rate)(x)
​
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate=drop_out_rate)(x)
​
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate=drop_out_rate)(x)
​
x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(rate=(drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting
​
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(1028, activation="relu")(x)
output_tensor = layers.Dense(number_of_classes, activation="softmax")(x)
​
model = tf.keras.Model(input_tensor, output_tensor)
model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=metrics)
Configuring model properties
model_filename = base_dir + "gunshot_sound_model.pkl"
​
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
#Optional debugging of the model's architecture
model.summary()
Training & caching the model
History = model.fit(train_wav, train_label,
          validation_data=[test_wav, test_label],
          epochs=number_of_epochs,
          callbacks=model_callbacks,
          verbose=1,
          batch_size=batch_size,
          shuffle=True)
​
model.save(base_dir + "gunshot_sound_model.h5")
#Summarizing history for accuracy
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#Summarizing history for loss
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#Optional debugging of incorrectly-labeled examples
y_test_pred = model.predict(test_wav)
y_predicted_classes_test = y_test_pred.argmax(axis=-1)
y_actual_classes_test= test_label.argmax(axis=-1)
wrong_examples = np.nonzero(y_predicted_classes_test != y_actual_classes_test)
print(wrong_examples)
#Optional debugging of an individual incorrectly-labeled example
i = 0
sample = np.reshape(test_wav[i], sample_rate_per_two_seconds, )
sample_rate = 22050
print(y_actual_classes_test[i], y_predicted_classes_test[i])
ipd.Audio(sample, rate=sample_rate)
