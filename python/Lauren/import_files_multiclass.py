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
                    elif (np.max(abs(sample_slice)) < gunshot_frequency_threshold) && (prescribed_label == "gunshot"):
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

