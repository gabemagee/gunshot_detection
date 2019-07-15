import numpy as np
import librosa

sampling_rate = 44100
hop_length = 345 * 2
fmin = 20
fmax = sampling_rate // 2
n_mels = 128
n_fft = n_mels * 20
min_seconds = 0.5

def make_spectrogram(y):
    y = np.array(y)
    print(type(y))
    print(y.dtype)
    return np.array(librosa.feature.melspectrogram(y=y, sr=22050))

def audio_to_melspectrogram(audio):
    audio = np.array(audio)
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=sampling_rate,
                                                 n_mels=n_mels,
                                                 hop_length=hop_length,
                                                 n_fft=n_fft,
                                                 fmin=fmin,
                                                 fmax=fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


base_dir = "/home/gamagee/workspace/gunshot_detection/REU_Data/spectrogram_training/"
src_np = base_dir + "gunshot_augmented_sound_samples.npy"

samples = np.array(np.load(src_np))

s = []
i = 0
for sample in samples:
    print(i)
    i = i + 1
    s.append(audio_to_melspectrogram(sample))

s = np.array(s)

np.save(base_dir + "gunshot_augmented_sound_samples_spectrogram_v2.npy",s)
