import os
import csv
import random
import librosa
import numpy as np

sample_rate_per_two_seconds = 44100

base_dir = "/home/gamagee/workspace/processing/"

samples_dir = base_dir+"Samples/"

csv_list = base_dir+"labels.csv"

current_index = 0

rows_to_append = []
labels = []
d = {}

n = 0
with open(csv_list) as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if row[0]=="ID":
            pass
        id = row[0]
        label = row[1]
        d[id] = label

samples = os.listdir(samples_dir)

for s in samples:
    s = s.split(".")[0]
    lbl = d[s]
    labels.append(lbl)

n = len(samples)

n_training = int(n*0.8)
n_testing = int(n*0.2)

n_validation = int(n_training*0.2)
n_training = int(n_training*0.8)
print(n)

training_set = []
training_labels = []
for i in range(n_training):
    j = random.randint(0,len(samples)-1)
    sample = samples.pop(j)
    sample_id = sample.split(".")[0]
    label = d[sample_id]
    training_labels.append(label)
    training_set.append(sample)

validation_set = []
validation_labels = []
for i in range(n_validation):
    j = random.randint(0,len(samples)-1)
    sample = samples.pop(j)
    sample_id = sample.split(".")[0]
    label = d[sample_id]
    validation_labels.append(label)
    validation_set.append(sample)

testing_set = []
testing_labels = []
for i in range(n_testing):
    j = random.randint(0,len(samples)-1)
    sample = samples.pop(j)
    sample_id = sample.split(".")[0]
    label = d[sample_id]
    testing_labels.append(label)
    testing_set.append(sample)

print(len(testing_set),len(training_set),len(validation_set))

samples = os.listdir(samples_dir)

sets = [samples,testing_set,validation_set,training_set,samples]
label_sets = [labels,testing_labels,validation_labels,training_labels,labels]
nombre = ["all","testing","validation","training","all"]

label_count = {}

for i in range(1):
    print("Starting parsing set of:",nombre[i])
    set = sets[i]
    label_set = label_sets[i]
    samples_processed = []
    labels_processed = []
    for j in range(len(set)):
        sample_file = set[j]
        label = label_set[j]
        sample, sample_rate = librosa.load(samples_dir + sample_file)

        if len(sample) <= sample_rate_per_two_seconds:
            number_of_missing_hertz = sample_rate_per_two_seconds - len(sample)
            padded_sample = list(np.array(sample.tolist() + [0 for i in range(number_of_missing_hertz)]))
            samples_processed.append(padded_sample)
            labels_processed.append(label)
            label_count[label] = label_count.get(label,0)+1
        else:
            for n in range(0, sample.size - sample_rate_per_two_seconds, sample_rate_per_two_seconds):
                sample_slice = list(sample[n: n + sample_rate_per_two_seconds])
                samples_processed.append(sample_slice)
                labels_processed.append(label)
                label_count[label] = label_count.get(label,0)+1

    filename = base_dir+nombre[i]
    samples_processed = np.array(samples_processed)
    print(samples_processed.shape)
    np.save(filename+"_samples.npy",samples_processed)
    np.save(filename+"_labels.npy",np.array(labels_processed))
    print("Finished parsing set of: ",nombre[i])

n = len(labels_processed)
print(n)
for label in label_count.keys():
    print(label,label_count[label]/n)
