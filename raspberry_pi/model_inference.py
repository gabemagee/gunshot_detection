
# coding: utf-8

# ## Package Imports

# In[ ]:

import pyaudio
import wave
import numpy as np


# ## Variable Initializations

# In[ ]:

audio_format = pyaudio.paFloat32
audio_rate = 44100
audio_channels = 1
audio_device_index = 1
audio_frames_per_buffer = 4096
audio_sample_duration = 2


# ## Processing Microphone Audio

# In[ ]:

pa = pyaudio.PyAudio()
    
stream = pa.open(format = audio_format,
                 rate = audio_rate,
                 channels = audio_channels,
                 input_device_index = audio_device_index,
                 frames_per_buffer = audio_frames_per_buffer,
                 input = True)

print("--- Recording Audio ---")
np_array_data = []

# Loops through the stream and appends audio chunks to the frame array
for i in range(0, int((audio_rate / audio_frames_per_buffer) * audio_sample_duration)):
    data = stream.read(audio_frames_per_buffer, exception_on_overflow = False)
    np_array_data.append(np.frombuffer(data, dtype=np.float32))
    
microphone_data = np.concatenate(np_array_data)
print("--- Finished Recording Audio ---")

# Stops the stream, closes it, and terminates the PyAudio instance
stream.stop_stream()
stream.close()
pa.terminate()

# Saves the audio frames as WAV files
wavefile = wave.open("mic_test",'wb')
wavefile.setnchannels(audio_channels)
wavefile.setsampwidth(pa.get_sample_size(audio_format))
wavefile.setframerate(audio_rate)
wavefile.writeframes(b''.join(microphone_data))
wavefile.close()

