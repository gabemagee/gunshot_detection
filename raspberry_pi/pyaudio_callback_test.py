#!/usr/bin/env python
# coding: utf-8

# Package Imports


import pyaudio
import time
import numpy as np
from queue import Queue

pa = pyaudio.PyAudio()
sound_data = np.zeros(0, dtype = "int16")
callback_queue = Queue()
i = 1

def callback(in_data, frame_count, time_info, status):
    global sound_data, i
    data = np.frombuffer(in_data, dtype = "int16")
    sound_data = np.append(sound_data, data)
    print("i'th iteration:", i, len(data))
    i += 1
    if len(sound_data) > 88200:
        callback_queue.put(sound_data)
        sound_data = np.zeros(88200, dtype = "int16")
        return (data, pyaudio.paComplete)
    return (data, pyaudio.paContinue)

stream = pa.open(format = pyaudio.paInt16, input_device_index = 5, channels = 1, rate = 44100, input = True, stream_callback = callback)

stream.start_stream()

while stream.is_active():
    time.sleep(0.1)

two_second_sample = callback_queue.get()
print(two_second_sample.shape)

stream.stop_stream()
stream.close()
pa.terminate()