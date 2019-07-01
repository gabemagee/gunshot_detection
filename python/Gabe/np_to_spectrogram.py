import numpy as np


def make_spectrogram(y):
    y = np.array(y)
    print(type(y))
    print(y.dtype)
    return np.array(librosa.feature.melspectrogram(y=y, sr=22050))


base_dir = "/home/gamagee/workspace/gunshot_detection/"
src_np = base_dir + "gunshot_augmented_sound_samples.npy"

samples = np.array(np.load(src_np))

s = []
for sample in samples:
    s.append(make_spectrogram(sample))

s = np.array(s)

np.save(base_dir + "gunshot_augmented_sound_samples_spectrogram.npy",s)
