from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import tensorflow as tf


def GetModel(input_shape):
    input_layer = Input((input_shape,))
    encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoded = Dense(50, activation='relu')(encoded)

    ## decoding part
    decoded = Dense(50, activation='tanh')(encoded)
    decoded = Dense(100, activation='tanh')(decoded)

    ## output layer
    output_layer = Dense(input_shape, activation='relu')(decoded)
    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adadelta", loss="mse")

    return autoencoder


if __name__ == '__main__':
    data = pd.read_csv('./data/dataset_v1.csv')
    X_train = data['url']
    vectorizer = CountVectorizer()

    X_train = vectorizer.fit_transform(X_train)

    if 'autoencoder.h5' in os.listdir('./saved_model/'):
        autoencoder = tf.keras.models.load_model('./saved_model/autoencoder.h5')
    else:
        autoencoder = GetModel(input_shape=X_train.shape[1])
        # Use unlabeled data to train the Encoder
        autoencoder.fit(X_train, X_train,
                        batch_size=256, epochs=10, verbose=1,
                        shuffle=True, validation_split=0.20)
        autoencoder.save('./saved_model/autoencoder.h5')

    hidden_representation = Sequential()
    hidden_representation.add(autoencoder.layers[0])
    hidden_representation.add(autoencoder.layers[1])
    hidden_representation.add(autoencoder.layers[2])

    data_labeled = pd.read_csv('./data/dataset_labeled.csv')
    X_labeled = X_train[0:799]
    # X_labeled = vectorizer.fit_transform(X_labeled)
    y_rep = data_labeled['NewCategory']
    X_rep = hidden_representation.predict(X_labeled)
    train_x, val_x, train_y, val_y = train_test_split(X_rep, y_rep, test_size=0.25)

    clf = LogisticRegression(solver="lbfgs").fit(train_x, train_y)
    pred_y = clf.predict(val_x)
    print("")
    print("Classification Report: ")
    print(classification_report(val_y, pred_y))

    print("")
    print("Accuracy Score: ", accuracy_score(val_y, pred_y))
