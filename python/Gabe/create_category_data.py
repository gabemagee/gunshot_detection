import os
import csv
import numpy as np

base = "/home/gamagee/workspace/gunshot_detection/REU_Data/REU_Samples_and_Labels/"

label_csv = base + "labels.csv"

samples = base + "Samples/"


with open(label_csv) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print(row)
