#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#imports 
import os

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import librosa
import soundfile
import re
import cv2
from sklearn.model_selection import KFold

import IPython.display as ipd


# In[ ]:


samples=[]
labels = []
gunshot_frequency_threshold = 0.25
sample_rate = 22050
sample_rate_per_two_seconds = 44100
input_shape = (sample_rate_per_two_seconds, 1)
base_dir = "/home/lauogden/data/"
data_dir = base_dir + "REU_Samples_and_Labels/"
sound_data_dir = data_dir + "Samples/"


# In[ ]:


#read in csv file with labels
sound_types = pd.read_csv(data_dir + "labels.csv")


# In[ ]:


#read in all the wav files
#MULTICLASS CLASSIFICATION:
    #0: miscellanious
    #1: gunshot
    #2: fireworks
    #3: glassbreak
    
print("...Parsing sound data...")
sound_file_id = 0
sound_file_names = []

for file in os.listdir(sound_data_dir):
    if file.endswith(".wav"):
        try:
            #get label from the csv
            sound_file_id = int(re.search(r'\d+', file).group())
            sample, sample_rate = librosa.load(sound_data_dir + file)
            prescribed_label = sound_types.loc[sound_types["ID"] == sound_file_id, "Class"].values[0]

            #for 2s or less long clips
            if len(sample) <= sample_rate_per_two_seconds:
                label = 0
                number_of_missing_hertz = sample_rate_per_two_seconds - len(sample)
                padded_sample = np.array(sample.tolist() + [0 for i in range(number_of_missing_hertz)])
                if prescribed_label == "gun_shot":
                    label = 1
                elif prescribed_label == "fireworks":
                    label = 2
                elif prescribed_label == "glassbreak":
                    label = 3

                #append the sample and label 
                samples.append(padded_sample)
                labels.append(label)
                sound_file_names.append(file)
            
            #for clips longer than 2s: split them up
            else:
                for i in range(0, sample.size - sample_rate_per_two_seconds, sample_rate_per_two_seconds):
                    sample_slice = sample[i : i + sample_rate_per_two_seconds]
                    label = 0
                    if prescribed_label == "gun_shot":
                        label = 1
                    elif prescribed_label == "fireworks":
                        label = 2
                    elif prescribed_label == "glassbreak":
                        label = 3
                    elif (np.max(abs(sample_slice)) < gunshot_frequency_threshold) and (prescribed_label == "gunshot"):
                        #check for silence to not mislabel gunshots
                        #maybe I should do this for fireworks and glassbreak also?
                        label = 0

                    #append the sample slice and label
                    samples.append(sample_slice)
                    labels.append(label)
                    sound_file_names.append(file)
        except:
            #if Librosa can't read a file or if something errs out
            print("Sound(s) not recognized by Librosa:" + file)
            pass


# In[ ]:


#print number of samples available
print("The number of samples available for training is currently " + str(len(samples)) + '.')
print("The number of labels available for training is currently " + str(len(labels)) + '.')


# In[ ]:


#save the samples and labels as numpy arrays
np.save(base_dir + "gunshot_sound_samples_multiclass.npy", samples)
np.save(base_dir + "gunshot_sound_labels_multiclass.npy", labels)




#AUGMENTATION

print("...moving to augmentation...")

# In[15]:


#data augmentation functions

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


# In[ ]:


#augment the data

samples = np.array(samples)
labels = np.array(labels)
number_of_augmentations = 5
augmented_samples = np.zeros((samples.shape[0] * (number_of_augmentations + 1), samples.shape[1]))
augmented_labels = np.zeros((labels.shape[0] * (number_of_augmentations + 1),))
j = 0

for i in range (0, len(augmented_samples), (number_of_augmentations + 1)):
    file = sound_file_names[j]
    
    augmented_samples[i,:] = samples[j,:]
    augmented_samples[i + 1,:] = time_shift(samples[j,:])
    augmented_samples[i + 2,:] = change_pitch(samples[j,:], sample_rate)
    augmented_samples[i + 3,:] = speed_change(samples[j,:])
    augmented_samples[i + 4,:] = change_volume(samples[j,:], np.random.uniform())
    if labels[j] == 1:
        augmented_samples[i + 5,:] = add_background(samples[j,:], file, data_dir, "") 
    else:
        augmented_samples[i + 5,:] = add_background(samples[j,:], file, data_dir, "gun_shot")
    
    augmented_labels[i] = labels[j]
    augmented_labels[i + 1] = labels[j]
    augmented_labels[i + 2] = labels[j]
    augmented_labels[i + 3] = labels[j]
    augmented_labels[i + 4] = labels[j]
    augmented_labels[i + 5] = labels[j]
    j += 1

samples = augmented_samples
labels = augmented_labels


# In[ ]:


#print number of samples
print("The number of samples available for training is currently " + str(len(samples)) + '.')
print("The number of labels available for training is currently " + str(len(labels)) + '.')


# In[ ]:


#save the augmented
np.save(base_dir + "gunshot_augmented_sound_samples_multiclass.npy", samples)
np.save(base_dir + "gunshot_augmented_sound_labels_multiclass.npy", labels)




