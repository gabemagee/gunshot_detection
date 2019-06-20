
# coding: utf-8

# ## Package Imports

# In[ ]:

import keras
import pyaudio


# ## Variable Initializations

# In[ ]:

audio_format = pyaudio.paInt16
audio_channels = 1
audio_rate = 44100
audio_device_index = 1
audio_input_block_time = 0.05
audio_input_frames_per_block = int(audio_rate * audio_input_block_time)


# ## Processing Microphone Audio

# In[ ]:

pa = pyaudio.PyAudio()
stream = pa.open(format = audio_format,
                 channels = audio_channels,
                 rate = audio_rate,
                 input = True,
                 input_device_index = audio_device_index,
                 frames_per_buffer = audio_input_frames_per_block)

while (True):
    try:
        block = stream.read(audio_input_frames_per_block)
    except IOerror as e:
        print("--- Error Trying to Process Microphone Audio ---")

