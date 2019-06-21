
# coding: utf-8

# ## Package Imports

# In[ ]:

import wave
import sounddevice as sd


# ## Variable Initializations

# In[ ]:

sample_rate = 44100
sample_duration = 2
number_of_channels = 1
sd.default.samplerate = sample_rate
sd.default.channels = number_of_channels


# ## Processing Microphone Audio

# In[ ]:

print("Now recording audio...")

sound_data_array = sd.rec(int(sample_duration * sample_rate))
sd.wait()

print("Finished recording audio...")

print(sound_data_array)

# save the audio frames as .wav file
wavefile = wave.open("test",'wb')
wavefile.setnchannels(number_of_channels)
wavefile.setsampwidth(4)
wavefile.setframerate(sample_rate)
wavefile.writeframes(b''.join(sound_data_array))
wavefile.close()
