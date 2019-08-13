import numpy as np

labels = "/Users/gabe/Downloads/models/processing/testing_labels.npy"
labels = np.load(labels)
print(labels.shape)
for label in labels:
    print(label)
