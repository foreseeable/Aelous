import pandas as pd
import numpy as np
import math


def is_nan(x):
    return x is np.nan or x != x


data = pd.read_csv('./data/dataset.csv')
data.sum()
category = data['categories']

# X_train = data['url']
# y_train = data['Categories']



def find_category():
    allcate = set()
    for id in np.arange(len(category)):
        if not is_nan(category[id]):
            print(category[id])
            rates = category[id].split(',')
            for item in rates:
                allcate.add(item)

    print(allcate)


def label_engineering():
    cate = ['media_introduction', 'others', 'location', 'social_media_profile', 'encyclopedia', \
            'qa_forum', 'shopping_item', 'list', 'media_player', 'article', 'NewCategory']
    proba = pd.DataFrame(index=np.arange(len(data)), columns=cate)
    for id in np.arange(len(category)):
        if not is_nan(category[id]):
            rates = category[id].split(',')
            cnt = {'media_introduction': 0, 'others': 0, 'location': 0, 'social_media_profile': 0, 'encyclopedia': 0, \
                   'qa_forum': 0, 'shopping_item': 0, 'list': 0, 'media_player': 0, 'article': 0}
            for item in rates:
                cnt[item] += 1
            value = list()
            for item in cnt:
                value.append(cnt[item])
            for item in cnt:
                if cnt[item] == np.max(value):
                    max_item = item
                    break
            for item in cnt:
                proba[item][id] = cnt[item] / len(rates)
                proba['NewCategory'][id] = max_item

    proba.sum()
    result = pd.concat([data, proba], axis=1, sort=False)
    result.sum()
    # print(result)
    result.to_csv('./data/dataset_v1.csv', index=False)


label_engineering()
