import PIL
import librosa
import numpy as np
#from librosa.core import power_to_db
# libraries
import matplotlib.pyplot as plt
import numpy as np
import csv




def numpy_to_image(numpy_array):
    sz = numpy_array.shape
    img = PIL.Image.new(mode = "L",size=sz)
    img.putdata(numpy_array)
    return img


#example = np.random.randint(255, size=(100, 100))
#print(example)
#img = numpy_to_image(example)
#img.show()


sampling_rate = 44100
hop_length = 345 * 2
fmin = 20
fmax = sampling_rate // 2
n_mels = 128
n_fft = n_mels * 20
min_seconds = 0.5

def make_spectrogram(y):
    y = np.array(y)
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

csv_list = "../processing/labels.csv"

gun_shot_ids = []

firework_ids = []

other_ids = []

jackhammer_ids = []

with open(csv_list) as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if row[0]=="ID":
            pass
        id = row[0]
        label = row[1]
        if label=="gun_shot":
            gun_shot_ids.append(id)
        if label=="jackhammer":
            jackhammer_ids.append(id)
        if label=="fireworks":
            firework_ids.append(id)
        else:
            other_ids.append(id)

def file_to_image(filename,class_name):
    id = filename.split(".")[0]
    y,sr = librosa.load(filename)
    spec_1 = make_spectrogram(y)
    spec_2 = audio_to_melspectrogram(y)

    plt.imshow(spec_2)
    plt.savefig("../processing/"+class_name+"_reg.png")
    plt.close()

    plt.imshow(spec_1)
    plt.savefig("../processing/"+class_name+"_mel.png")
    plt.close()


nom = ["gunshot","fireworks","other","jackhammer"]
lists = [gun_shot_ids, firework_ids,other_ids,jackhammer_ids]

import random

for i in range(4):
    lst = lists[i]
    file_path = "../processing/Samples/"+lst[random.randint(0,len(lst)-1)]+".wav"
    file_to_image(file_path,nom[i])
