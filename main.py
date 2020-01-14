import numpy as np
import pandas as pd
import tensorflow as tf


def main():
    data = pd.read_csv('./data/dataset_v1.csv')
    X_train = data['url']
    y_train = data['NewCategory']


if __name__ == '__main__':
    main()
