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
import cv2
import six
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ### Graph Libraries

# In [ ]:

import matplotlib.pyplot as plt

# # Initialization of Variables

# In[ ]:


GUNSHOT_FREQUENCY_THESHOLD = 0.25
SAMPLE_RATE_PER_SECOND = 22050
SAMPLE_RATE_PER_TWO_SECONDS = 44100
SOUND_FILE_ID = 0
MAXIMUM_AUDIO_FRAME_INTEGER_VALUE = 2 ** 15 - 1
SOUND_NORMALIZATION_THRESHOLD = 10 ** (-1.0 / 20)
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


def normalize(sound_data):
    normalization_factor = float(SOUND_NORMALIZATION_THRESHOLD * MAXIMUM_AUDIO_FRAME_INTEGER_VALUE) / max(abs(i) for i in sound_data)
    
    # Averages the volume out
    r = array('f')
    for datum in sound_data:
        r.append(int(datum * normalization_factor))
    return np.array(r, dtype = np.float32)


def convert_to_spectrogram(data, sample_rate):
    return np.array(librosa.feature.melspectrogram(y=data, sr=sample_rate), dtype="float32")


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    S = np.asarray(S)
    if amin <= 0:
        logger.debug('ParameterError: amin must be strictly positive')
    if np.issubdtype(S.dtype, np.complexfloating):
        logger.debug('Warning: power_to_db was called on complex input so phase '
                     'information will be discarded. To suppress this warning, '
                     'call power_to_db(np.abs(D)**2) instead.')
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
            logger.debug('ParameterError: top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def convert_spectrogram_to_image(spectrogram):
    plt.interactive(False)

    figure = plt.figure(figsize=[0.72, 0.72], dpi=400)
    plt.tight_layout(pad=0)
    ax = figure.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    librosa.display.specshow(power_to_db(spectrogram, ref=np.max))

    canvas = FigureCanvas(figure)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    image = np.fromstring(figure.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape((width, height, 3))
    image = cv2.resize(image, (192, 192))

    # Cleaning up the matplotlib instance
    plt.close()
    figure.clf()
    plt.close(figure)
    plt.close("all")

    # Returns a NumPy array containing an image of a spectrogram
    return image


# ### Iteratively Converting All Augmented Samples into Spectrograms

# In[ ]:


spectrograms = []

for sample in samples:
    sample = normalize(sample)  # Normalizes augmented samples after loading them
    spectrogram = convert_to_spectrogram(sample, SAMPLE_RATE_PER_SECOND)
    spectrogram = convert_spectrogram_to_image(spectrogram)
    spectrograms.append(spectrogram)
    print("Converted a sample into a spectrogram...")

# ### Restructuring spectrograms

# In[ ]:


samples = np.array(spectrograms).reshape(-1, 192, 192, 3)
samples = samples.astype("float32")
samples /= 255
print("Finished loading all spectrograms into memory...")

# ## Saving spectrograms as a NumPy array

# In[ ]:


np.save(BASE_DIRECTORY + "gunshot_augmented_sample_spectrograms.npy", samples)
print("Successfully saved all spectrograms as a NumPy array...")
