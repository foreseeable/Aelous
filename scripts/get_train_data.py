import csv
import random
import pickle

data = []
with open("/home/wcmg_chen/ml-camp/webpage-classification/dataset.csv") as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        if row['is entity']=='0' or row['is entity']=='1':
            data.append(row)

random.shuffle(data)
N = len(data)
P = int(N*0.9)
train = data[:P]
test = data[P:]
with open('data.pickle', 'wb') as f:
        pickle.dump((train,test),f)


