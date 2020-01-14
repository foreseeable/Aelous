# this is our input placeholder

import os
import numpy as np
import pandas as pd
from time import time
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from tensorflow.keras.layers import Dense
import tensorflow as tf
from os.path import splitext
import ipaddress as ip
from urllib.parse import urlparse
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('./data/dataset_v1.csv')
X_train = data['url']
y_train = data['NewCategory']
vectorizer = CountVectorizer()


# X = vectorizer.fit_transform(X_train)

def NaiveBayes():
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf = text_clf.fit(X_train, y_train)
    # grid_mean_scores = [result.mean_validation_score for result in gs_clf.grid_scores_]
    # print(grid_mean_scores)
    y_pred = text_clf.predict(X_train)
    prediction = pd.DataFrame(y_pred, index=np.arange(len(data)), columns=['prediction'])
    result = pd.concat([X_train, prediction], axis=1, sort=False)
    result.sum()
    result.to_csv('./pred/pred1.csv', index=False)
    precision_recall_fscore_support(y_train, y_pred, average='weighted')
    print("")
    print("Classification Report: ")
    print(classification_report(y_train, y_pred))
    print("")
    print("Accuracy Score: ", accuracy_score(y_train, y_pred))


def NaiveKmeans():
    km = KMeans(n_jobs=-1, n_clusters=10, n_init=20)
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', km)])
    text_clf.fit(X_train)
    y_pred = text_clf.predict(X_train)
    prediction = pd.DataFrame(y_pred, index=np.arange(len(data)), columns=['prediction'])
    print(prediction)
    prediction.describe()
    result = pd.concat([X_train, prediction], axis=1, sort=False)
    result.to_csv('./pred/pred2.csv', index=False)


# NaiveBayes()
NaiveKmeans()
