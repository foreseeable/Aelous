"""
Author: Yujie Lu

crop.py: Used to crop the image to square shape
"""
import pandas as pd
import numpy as np

train = pd.read_csv('./data/dataset_v1.csv')

print('/home/yujie6cs/webpage-classification/' + train['render'][0])

from matplotlib import pyplot as plt

for id in np.arange(len(train)):
    f = plt.imread('/home/yujie6cs/webpage-classification/' + train['render'][id])
    if f.shape[0] > f.shape[1]:
        f = f[0:f.shape[1], 0:f.shape[1], :]
        # print(x['url'],f.shape)
    plt.imsave('/home/yujie6cs/' + train['render'][id], f)
    if id % 500 == 0:
        print("Cropped image numbers:", id)
