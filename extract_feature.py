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


def countdots(url):
    return url.count('.')


def countdelim(url):
    count = 0
    delim = [';', '_', '?', '=', '&']
    for each in url:
        if each in delim:
            count = count + 1

    return count


# method to check the presence of hyphens

def isPresentHyphen(url):
    return url.count('-')


def isPresentDSlash(url):
    return url.count('//')


def get_ext(url):
    """Return the filename extension from url, or ''."""
    root, ext = splitext(url)
    return ext


def countSubDir(url):
    return url.count('/')


def countSubDomain(subdomain):
    if not subdomain:
        return 0
    else:
        return len(subdomain.split('.'))


def countQueries(query):
    if not query:
        return 0
    else:
        return len(query.split('&'))


from urllib.parse import urlparse
import tldextract


def getFeatures(url, label):
    result = []
    url = str(url)
    # add the url to feature set
    result.append(url)
    # parse the URL and extract the domain information
    path = urlparse(url)
    ext = tldextract.extract(url)
    # counting number of dots in subdomain
    result.append(countdots(ext.subdomain))
    # checking hyphen in domain
    result.append(isPresentHyphen(path.netloc))
    # length of URL
    result.append(len(url))
    # checking presence of double slash
    result.append(isPresentDSlash(path.path))
    # Count number of subdir
    result.append(countSubDir(path.path))
    # number of sub domain
    result.append(countSubDomain(ext.subdomain))
    # length of domain name
    result.append(len(path.netloc))
    # count number of queries
    result.append(len(path.query))
    return result