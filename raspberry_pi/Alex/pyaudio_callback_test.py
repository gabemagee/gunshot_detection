#!/usr/bin/env python
# coding: utf-8

# Package Imports


import pyaudio
import numpy as np
from queue import Queue

pa = pyaudio.PyAudio()
sound_data = np.zeros(0, dtype="int16")
callback_queue = Queue()
i = 1


def callback(in_data, frame_count, time_info, status):
    global sound_data, i
    data = np.frombuffer(in_data, dtype="int16")
    sound_data = np.append(sound_data, data)
    print("i'th iteration:", i, len(data))
    i += 1
    if len(sound_data) >= 88200:
        callback_queue.put(sound_data)
        sound_data = np.zeros(0, dtype="int16")
        i = 1
    return data, pyaudio.paContinue


stream = pa.open(format=pyaudio.paInt16, input_device_index=6, channels=1, rate=44100, input=True,
                 frames_per_buffer=4410, stream_callback=callback)

stream.start_stream()

while True:
    two_second_sample = callback_queue.get()
    print(two_second_sample.shape)
