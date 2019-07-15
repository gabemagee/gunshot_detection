#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# ### Math Libraries

# In[ ]:


import numpy as np

# ### Data Pre-Processing Libraries

# In[ ]:


import librosa
import six

# # Initialization of Variables

# In[ ]:


GUNSHOT_FREQUENCY_THESHOLD = 0.25
SAMPLE_RATE_PER_SECOND = 22050
SAMPLE_RATE_PER_TWO_SECONDS = 44100
HOP_LENGTH = 345 * 2
MINIMUM_FREQUENCY = 20
MAXIMUM_FREQUENCY = SAMPLE_RATE_PER_SECOND
NUMBER_OF_MELS = 128
NUMBER_OF_FFTS = NUMBER_OF_MELS * 20
BASE_DIRECTORY = "/home/amorehe/Datasets/"
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


def convert_audio_to_spectrogram(data):
    spectrogram = librosa.feature.melspectrogram(y=data, sr=SAMPLE_RATE_PER_TWO_SECONDS,
                                                 hop_length=HOP_LENGTH,
                                                 fmin=MINIMUM_FREQUENCY,
                                                 fmax=MAXIMUM_FREQUENCY,
                                                 n_mels=NUMBER_OF_MELS,
                                                 n_fft=NUMBER_OF_FFTS)
    spectrogram = power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    S = np.asarray(S)
    if amin <= 0:
        logger.debug("ParameterError: amin must be strictly positive")
    if np.issubdtype(S.dtype, np.complexfloating):
        logger.debug("Warning: power_to_db was called on complex input so phase information will be discarded.")
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
            logger.debug("ParameterError: top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


# ### Iteratively Converting All Augmented Samples into Spectrograms

# In[ ]:


spectrograms = []

for sample in samples:
    spectrogram = convert_audio_to_spectrogram(sample)
    spectrograms.append(spectrogram)
    print("Converted a sample into a spectrogram...")

# ### Restructuring spectrograms

# In[ ]:


samples = np.array(spectrograms).reshape(-1, 128, 84, 1)
samples = samples.astype("float32")
print("Finished loading all spectrograms into memory...")

# ## Saving spectrograms as a NumPy array

# In[ ]:


np.save(BASE_DIRECTORY + "gunshot_augmented_sample_spectrograms.npy", samples)
print("Successfully saved all spectrograms as a NumPy array...")
