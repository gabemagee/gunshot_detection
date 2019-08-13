#!/usr/bin/env python
# coding: utf-8

# ## Package Imports

# In[ ]:


import pyaudio
import librosa
import logging
import time
import numpy as np
from queue import Queue


# ## Configuring the Logger

# In[ ]:


logger = logging.getLogger('debugger')
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler('output.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# ## Variable Initializations

# In[ ]:


AUDIO_FORMAT = pyaudio.paFloat32
AUDIO_RATE = 44100
NUMBER_OF_AUDIO_CHANNELS = 1
AUDIO_DEVICE_INDEX = 6
NUMBER_OF_FRAMES_PER_BUFFER = 4410
SAMPLE_DURATION = 2
sound_data = np.zeros(0, dtype = "float32")
audio_sample_counter = 1
audio_capture_queue = Queue()


# ### WAV File Composition Function

# In[ ]:


# Saves a two-second audio sample as a WAV file
def create_wav_file(microphone_data, index, timestamp, model_used = ""):
    librosa.output.write_wav("/home/alexm/Audio Capture System Recordings/Audio Sample #"
                            + str(index) + " ("
                            + str(timestamp) + ").wav", microphone_data, 22050)


# ## Defining Threads

# ### Callback Thread

# In[ ]:


def callback(in_data, frame_count, time_info, status):
    global sound_data
    sound_buffer = np.frombuffer(in_data, dtype = "float32")
    sound_data = np.append(sound_data, sound_buffer)
    if len(sound_data) >= AUDIO_RATE * 2:
        audio_capture_queue.put(sound_data)
        current_time = time.ctime(time.time())
        audio_capture_queue.put(current_time)
        sound_data = np.zeros(0, dtype = "float32")
    return sound_buffer, pyaudio.paContinue

pa = pyaudio.PyAudio()

stream = pa.open(format = AUDIO_FORMAT,
                 rate = AUDIO_RATE,
                 channels = NUMBER_OF_AUDIO_CHANNELS,
                 input_device_index = AUDIO_DEVICE_INDEX,
                 frames_per_buffer = NUMBER_OF_FRAMES_PER_BUFFER,
                 input = True,
                 stream_callback = callback)

# Starts the callback thread
stream.start_stream()
logger.debug("--- Listening to Audio Stream ---")


# ### Main (Audio Capture) Thread

# In[ ]:


# This thread will run indefinitely
while True:
    # Gets a sample and its timestamp from the audio capture queue
    microphone_data = np.array(audio_capture_queue.get(), dtype = "float32")
    time_of_sample_occurrence = audio_capture_queue.get()
    
    # Cleans up the global NumPy audio data source
    sound_data = np.zeros(0, dtype = "float32")
        
    # Post-processes the microphone data
    modified_microphone_data = librosa.resample(y = microphone_data, orig_sr = AUDIO_RATE, target_sr = 22050)
    modified_microphone_data = modified_microphone_data[:44100]
            
    # Makes a WAV file of the audio sample
    create_wav_file(modified_microphone_data, audio_sample_counter, time_of_sample_occurrence)

    # Increments the counter for audio sample file names
    audio_sample_counter += 1


# In[ ]:




