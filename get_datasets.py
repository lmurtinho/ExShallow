import bz2
import numpy as np
import pandas as pd
import pickle
import requests
import re
import os
import shutil
import tarfile
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.datasets import load_iris, load_digits, load_svmlight_file
from sklearn.datasets import fetch_20newsgroups, fetch_openml, fetch_covtype
from sklearn.preprocessing import MinMaxScaler

def get_bz(name, url):
    bzfile = "/tmp/{}.bz".format(name)
    txtfile = "/tmp/{}.txt".format(name)

    r = requests.get(url, allow_redirects=True)

    with open(bzfile, "wb") as f:
        f.write(r.content)
    with open(bzfile, 'rb') as f:
        d = bz2.decompress(f.read())
    with open(txtfile, 'wb') as f:
        f.write(d)

    data = load_svmlight_file(txtfile)[0].toarray()

    os.remove(bzfile)
    os.remove(txtfile)
    return data

def get_anuran():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip'
    zipresp = requests.get(url, allow_redirects=True)
    with open("/tmp/tempfile.zip", "wb") as f:
        f.write(zipresp.content)
    zf = ZipFile("/tmp/tempfile.zip")
    zf.extractall(path='/tmp/')
    zf.close()
    data = pd.read_csv('/tmp/Frogs_MFCCs.csv').iloc[:,:22].values

    os.remove('/tmp/Frogs_MFCCs.csv')
    return data.astype(float), 10

def get_avila():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip'
    zipresp = requests.get(url, allow_redirects=True)
    with open("/tmp/tempfile.zip", "wb") as f:
        f.write(zipresp.content)
    zf = ZipFile("/tmp/tempfile.zip")
    zf.extractall(path='/tmp/')
    zf.close()

    with open('/tmp/avila/avila-tr.txt', 'r') as f:
        text = [l for l in f.readlines()]

    data_train = np.array([[float(j) for j in text[i].split(',')[:-1]]
                            for i in range(len(text))])

    with open('/tmp/avila/avila-ts.txt', 'r') as f:
        text = [l for l in f.readlines()]

    data_test = np.array([[float(j) for j in text[i].split(',')[:-1]]
                            for i in range(len(text))])

    data = np.concatenate((data_train, data_test))
    mms = MinMaxScaler()
    data = mms.fit_transform(data)
    shutil.rmtree('/tmp/avila')
    os.remove('/tmp/tempfile.zip')
    return np.array(data), 12

def get_beer():
    url = 'https://www.openml.org/data/download/21552938/dataset'
    r = requests.get(url, allow_redirects=True)
    lines = r.text.split('\n')
    floats = [re.findall(r'\d\.\d', l) for l in lines]
    data = np.array([[float(v) for v in f][:-1]
                     for f in floats
                     if len(f) == 6])
    return data, 104

def get_bng():
    url = 'https://www.openml.org/data/download/583865/BNG_audiology_1000_5.arff'
    r = requests.get(url, allow_redirects=True)
    lines = r.text.split('\n')
    vals = [l.split(',') for l in lines]
    vals = [v[:-1] for v in vals if len(v) == 70]

    data = pd.DataFrame(vals)
    data = pd.get_dummies(data, drop_first=True)
    data = np.array(data)
    return data, 24

def get_cifar10():
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    tar = requests.get(url, allow_redirects=True)
    with open("/tmp/temp.tar.gz", "wb") as f:
        f.write(tar.content)
    tar = tarfile.open('/tmp/temp.tar.gz', 'r:gz')
    tar.extractall(path='/tmp/')
    tar.close()

    with open('/tmp/cifar-10-batches-py/test_batch', 'rb') as f:
        data = pickle.load(f, encoding='bytes')[b'data']

    trains = [i for i in os.listdir('/tmp/cifar-10-batches-py')
                if 'data_batch' in i]
    for train in trains:
        file = '/tmp/cifar-10-batches-py/' + train
        with open(file, 'rb') as f:
            data = np.concatenate([data,
                                   pickle.load(f, encoding='bytes')[b'data']])
    shutil.rmtree('/tmp/cifar-10-batches-py')
    data = data.astype(np.float64)
    return data, 10

def get_collins():
    url = 'https://www.openml.org/data/get_csv/17953251/php5OMDBD'
    data = pd.read_csv(url).iloc[:,1:-4]
    data = np.array(data)
    mms = MinMaxScaler()
    data = mms.fit_transform(data)
    return data, 30

def get_covtype():
    return fetch_covtype().data, 7

def get_digits():
    return load_digits().data, 10

def get_iris():
    return load_iris().data, 3

# def get_letter():
#     url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale'
#     r = requests.get(url, allow_redirects=True)
#     with open('letter.txt', 'w') as f:
#         f.write(r.text)
#     sps = load_svmlight_file('letter.txt')
#     train = sps[0].toarray()
#
#     url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t'
#     r = requests.get(url, allow_redirects=True)
#     with open('letter.txt', 'w') as f:
#         f.write(r.text)
#     sps = load_svmlight_file('letter.txt')
#     test = sps[0].toarray()
#
#     data = np.concatenate([train, test])
#
#     os.remove('letter.txt')
#     return data, 26

def get_letter():
    data = fetch_openml('letter', version=1).data
    mms = MinMaxScaler()
    return mms.fit_transform(data), 26

def get_mice():
    data = fetch_openml(name="miceprotein", version=4).data
    return data[~np.isnan(data).any(axis=1)].values, 8

def get_newsgroups():
    dataset = fetch_20newsgroups(subset='all',
                                 remove=('headers', 'footers', 'quotes'),
                                 shuffle=True)
    vectorizer = TfidfVectorizer(stop_words='english',
                                 token_pattern=r'\b[^\d\W]+\b',
                                 min_df=.01, max_df=.1)
    return vectorizer.fit_transform(dataset.data).toarray(), 20

def get_pendigits():
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits'
    r = requests.get(url, allow_redirects=True)
    with open('pendigits.txt', 'w') as f:
        f.write(r.text)
    sps = load_svmlight_file('pendigits.txt')
    train = sps[0].toarray()

    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t'
    r = requests.get(url, allow_redirects=True)
    with open('pendigits.txt', 'w') as f:
        f.write(r.text)
    sps = load_svmlight_file('pendigits.txt')
    test = sps[0].toarray()

    data = np.concatenate([train, test])

    os.remove('pendigits.txt')
    return data, 10

def get_poker():
    url_train = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/poker.bz2'
    url_test = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/poker.t.bz2'
    name = 'poker'
    train = get_bz(name, url_train)
    test = get_bz(name, url_test)
    data = np.concatenate([train, test])
    return data, 10

def get_sensorless():
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/Sensorless.scale'
    r = requests.get(url, allow_redirects=True)
    with open('sensorless.txt', 'w') as f:
        f.write(r.text)
    sps = load_svmlight_file('sensorless.txt')
    data = sps[0].toarray()

    os.remove('sensorless.txt')
    return data, 11

def get_vowel():
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vowel.scale'
    r = requests.get(url, allow_redirects=True)
    with open('vowel.txt', 'w') as f:
        f.write(r.text)
    sps = load_svmlight_file('vowel.txt')
    train = sps[0].toarray()

    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vowel.scale.t'
    r = requests.get(url, allow_redirects=True)
    with open('vowel.txt', 'w') as f:
        f.write(r.text)
    sps = load_svmlight_file('vowel.txt')
    test = sps[0].toarray()

    data = np.concatenate([train, test])

    os.remove('vowel.txt')
    return data, 11
