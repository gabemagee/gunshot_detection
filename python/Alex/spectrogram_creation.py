#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# ### Math Libraries

# In[ ]:


import numpy as np

# ### Data Pre-Processing Libraries

# In[ ]:


import librosa
import librosa.display

# ### Graph Libraries

# In [ ]:

import matplotlib.pyplot as plt

# # Initialization of Variables

# In[ ]:


GUNSHOT_FREQUENCY_THESHOLD = 0.25
SAMPLE_RATE_PER_SECOND = 22050
SAMPLE_RATE_PER_TWO_SECONDS = 44100
SOUND_FILE_ID = 0
BASE_DIRECTORY = "/home/alexm/Datasets/"
DATA_DIRECTORY = BASE_DIRECTORY + "REU_Samples_and_Labels/"
SPECTROGRAM_DIRECTORY = BASE_DIRECTORY + "Spectrograms/"
SOUND_DATA_DIRECTORY = DATA_DIRECTORY + "Samples/"
samples = []
labels = []
sound_file_names = []
sample_weights = []

# # Data Pre-Processing


# ## Loading augmented sample file and label file as numpy arrays

# In[ ]:


samples = np.load(BASE_DIRECTORY + "gunshot_augmented_sound_samples.npy")
labels = np.load(BASE_DIRECTORY + "gunshot_augmented_sound_labels.npy")


# ## Converting Augmented Samples to Spectrograms

# ### Defining Spectrogram Conversion Functions

# In[ ]:


def convert_to_spectrogram(data, sample_rate):
    return np.array(librosa.feature.melspectrogram(y=data, sr=sample_rate))


def save_spectrogram_as_png(spectrogram, index):
    plt.interactive(False)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))
    plt.savefig("~/Datasets/Spectrograms/" + str(index) + ".png", dpi=400, bbox_inches="tight", pad_inches=0)

    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')


# ### Iteratively Converting All Augmented Samples into Spectrograms

# In[ ]:


spectogram_index = 0

for sample in samples:
    spectrogram = convert_to_spectrogram(sample, sample_rate)
    save_spectrogram_as_png(spectrogram, spectogram_index)
    print("Successfully saved augmented sample #" + str(spectogram_index) + " as a spectrogram...")
    spectogram_index += 1
