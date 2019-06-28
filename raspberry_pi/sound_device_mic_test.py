#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import sounddevice as sd
# import librosa


# In[ ]:


# sd.default.device = 6
# sd.default.samplerate = 44100
# sd.default.channels = 1


# In[ ]:


# rec = sd.rec(int(2 * 44100), dtype="int16", blocksize = 24000)
# sd.wait()
# librosa.output.write_wav("SoundDevice Output from Mic.wav", rec, sr=44100)


# ## New Attempt

# In[ ]:


import sys
import time
import queue

import sounddevice as sd
import soundfile as sf

RECORD_TIME = 2

q = queue.Queue()
rec_start = int(time.time())

sd.default.device = 7
sd.default.samplerate = 44100
sd.default.channels = 1

def data_callback(input_data, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(input_data.copy())

with sf.SoundFile('SoundDevice Output from Mic.wav', mode='x', samplerate=44100, channels=1) as file:
    with sd.InputStream(callback=data_callback, blocksize=24000, dtype="int16"):
        rec_time = int(time.time()) - rec_start
        
        while rec_time <= RECORD_TIME:
            data = q.get()
            print(data)
            file.write(data)
            rec_time = int(time.time()) - rec_start

