"""
Author: Yujie Lu

pre-process.py: Used to pre-process the dataset (the original dataset is './data/dataset.csv')
Here is the basic information of our dataset

- Dataset name: webpage-classification
- Data size: 10K pairs of {web page screenshot image, web page html source}.
- 800 samples are manually labeled {is_entity, category}
- 2590 samples are manually labeled {is_entity}

Specifically, there are 2 useful functions:
1. Rates2proba(): transform the raters' choices into probability of each category.
The generated dataset is saved as '/data/dataset_v1.csv'.
2. LabelByRule(): Use rule-based methods to label more data of the dataset by the
key words in the url. The labeled dataset is saved in '/data/more_label.csv'
3. RemoveNan(file): in order to do supervised learning, we have to remove
the data that is not labeled.
"""

import pandas as pd
import numpy as np
import math


def is_nan(x):
    """Helper function to label the data"""
    return x is np.nan or x != x


def CountHyphen(url):
    """Helper function to label the data"""
    return url.count('-')


def ResetProba(df, id):
    """Helper function to label the data"""
    df['media_introduction'][id] = 0.0
    df['others'][id] = 0.0
    df['location'][id] = 0.0
    df['social_media_profile'][id] = 0.0
    df['encyclopedia'][id] = 0.0
    df['qa_forum'][id] = 0.0
    df['shopping_item'][id] = 0.0
    df['list'][id] = 0.0
    df['media_player'][id] = 0.0
    df['article'][id] = 0.0


# X_train = data['url']
# y_train = data['Categories']


def check_category():
    data = pd.read_csv('./data/dataset.csv')
    data.sum()
    category = data['categories']
    allcate = set()
    for id in np.arange(len(category)):
        if not is_nan(category[id]):
            print(category[id])
            rates = category[id].split(',')
            for item in rates:
                allcate.add(item)

    print(allcate)


def RemoveNan(file):
    data_nan = pd.read_csv('../data/' + file + '.csv')
    data_nan.info()
    category = data_nan['NewCategory']
    for id in np.arange(len(data_nan)):
        if is_nan(category[id]):
            data_nan = data_nan.drop(id, axis=0)
            # print('remove id: ', id)
    print('drop all lines with nan for ', file)
    data_nan.to_csv('../data/' + file + '-FullLabel.csv')


def Rates2proba():
    data = pd.read_csv('./data/dataset.csv')
    data.sum()
    category = data['categories']

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


def LabelByRule():
    sum_label = 0
    data_v1 = pd.read_csv('./data/dataset_v1.csv')
    category = data_v1['categories']
    url = data_v1['url']
    import extract_feature
    for id in np.arange(len(data_v1)):
        if not is_nan(category[id]):
            sum_label += 1
        else:
            ResetProba(data_v1, id)
            # print('Processing the dataline: ', id)
            url_t = url[id]
            is_blog = CountHyphen(url_t) > 3
            if 'wikipeida' in url_t:
                data_v1['NewCategory'][id] = 'encyclopedia'
                data_v1['encyclopedia'][id] = 1.0
                print('Find one encyclopedia: ', url_t)
                sum_label += 1
            elif 'stackoverflow' in url_t or 'reddit' in url_t:
                data_v1['NewCategory'][id] = 'qa_forum'
                data_v1['qa_forum'][id] = 1.0
                print('Find one qa_forum: ', url_t)
                sum_label += 1
            elif 'profile' in url_t or 'scholar.google' in url_t:
                data_v1['NewCategory'][id] = 'social_media_profile'
                data_v1['social_media_profile'][id] = 1.0
                print('Find one social_media_profile: ', url_t)
                sum_label += 1

            if 'quora' in url_t:
                data_v1['NewCategory'][id] = 'qa_forum'
                data_v1['qa_forum'][id] = 1.0
                print('Find one qa_forum: ', url_t)
                sum_label += 1

            elif 'twitter' in url_t:
                data_v1['NewCategory'][id] = 'social_media_profile'
                data_v1['social_media_profile'][id] = 1.0
                print('Find one social_media_profile: ', url_t)
                sum_label += 1

            elif 'imdb' in url_t:
                if 'name' in url_t:
                    data_v1['NewCategory'][id] = 'social_media_profile'
                    data_v1['social_media_profile'][id] = 1.0
                    print('Find one social_media_profile: ', url_t)
                else:
                    data_v1['NewCategory'][id] = 'media_introduction'
                    data_v1['media_introduction'][id] = 1.0
                    print('Find one media_introduction: ', url_t)
                sum_label += 1

            elif CountHyphen(url_t) > 3:
                data_v1['NewCategory'][id] = 'article'
                data_v1['article'][id] = 0.7
                data_v1['qa_forum'][id] = 0.3
                sum_label += 1
                # print('Find one article: ', url_t)

    data_v1.to_csv('./data/more_label.csv')
    print('Labeled data in total: ', sum_label)


if __name__ == '__main__':
    LabelByRule()
