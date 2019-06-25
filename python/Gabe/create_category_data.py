import os
import csv
import numpy as np

base = "/home/gamagee/workspace/gunshot_detection/REU_Data/REU_Samples_and_Labels/"

label_csv = base + "labels.csv"

samples = base + "Samples/"

types = []

with open(label_csv) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row[1] not in types and row[1]!="Class":
            types.append(row[1])
