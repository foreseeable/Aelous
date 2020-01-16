import pandas as pd
import numpy as np
import math


def is_nan(x):
    return x is np.nan or x != x


def CountHyphen(url):
    return url.count('-')


def ResetProba(df, id):
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


def LabelByRule(sum_label=0):
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
    
def RemoveNan(file):
    data_nan = pd.read_csv('./data/' + file + '.csv')
    category = data_nan['categories']
    for id in np.arange(len(data_nan)):
        if not is_nan(category[id]):
            data_nan.drop(id, axis=0)
    print('drop all lines with nan for ', file)
    data_nan.to_csv('./data/' + file + '-FullLabel.csv')
    


if __name__ == '__main__':
    # LabelByRule()
    RemoveNan('more_label')

# label_engineering()
