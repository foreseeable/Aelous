import pandas as pd
import numpy as np

train = pd.read_csv('./data/dataset_v1.csv')

print('../../webpage-classification/' + train['render'][0])

from matplotlib import pyplot as plt

for id in np.arange(len(train)):
    f = plt.imread('../../webpage-classification/' + train['render'][id])
    if f.shape[0] > f.shape[1]:
        f = f[0:f.shape[1], 0:f.shape[1], :]
        # print(x['url'],f.shape)
    plt.imsave('$HOME/img/' + train['render'][id], f)
    # plt.imshow(f)
    # plt.show()
    # cv2.imshow(f)
    # print(f.shape)
