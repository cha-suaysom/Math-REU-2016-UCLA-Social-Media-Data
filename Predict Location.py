import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF  # , PCA, TruncatedSVD, LatentDirichletAllocation
import sklearn.preprocessing
import time
import re  # regex
import scipy.io as sio
import scipy.sparse as sps
import numpy as np
import math
from scipy.stats.mstats import mode
import pandas as pd

(W, H) = pickle.load(open('Location_NMF_100_topics_barc_WH.pkl','rb'))
rest_of_tweets_TFIDF  = pickle.load(open('rest_of_tweets_TFIDF_barc.pkl','rb'))
print(W.shape, H.shape)
Spatial_sample = pickle.load(open('Location_pandas_data_barc.pkl', 'rb'))
Topics = W.argmax(axis=1)
Spatial_sample["topics"] = Topics.tolist()
pickle.dump(Spatial_sample, open('Location_pandas_data_barc.pkl', 'wb'))

rest_of_tweets_Data = pickle.load(open('rest_of_tweets_pandas_data_barc.pkl','rb'))
normalized_H = sklearn.preprocessing.normalize(H[:,:-10000])
print(np.linalg.norm((normalized_H[0:2, :]), 'fro'))
print(normalized_H.shape,rest_of_tweets_TFIDF.shape)

normalized_tweets = sklearn.preprocessing.normalize(rest_of_tweets_TFIDF)

topics = normalized_H*(normalized_tweets.T)
pickle.dump(topics, open('test_topic_ditribution.pkl', 'wb'))
topics = topics.T
B = (np.max(topics, axis = 1)>0.2)


Topic_list = (np.argmax(topics, axis = 1)).tolist()
print(len(Topic_list))
print(len(rest_of_tweets_Data.index))
rest_of_tweets_Data["topics"]= Topic_list
pickle.dump(rest_of_tweets_Data, open('rest_of_tweets_pandas_data_barc.pkl','wb'))
Topic_stats = pickle.load(open('topic_stats_pandas.pkl', 'rb'))
onekmlist = []
twokmlist = []
selected_tweets_data = rest_of_tweets_Data[B]
for T in range(0,100):
    subset= selected_tweets_data[selected_tweets_data["topics"]== T]
    xgrid = subset["xgrid"].tolist()
    ygrid = subset["ygrid"].tolist()
    A = np.asarray((Topic_stats["peak"]))
    print(mode(xgrid, axis = None))
    print(mode(ygrid, axis = None))
    print((A[T])[0], (A[T])[1])
    length = len(xgrid)
    distance = 0
    oneKm = 0
    twoKm = 0
    for i in range(0,length):
        Dis = math.sqrt((xgrid[i]-A[T][0])**2 + (ygrid[i] -A[T][1])**2)
        if  Dis < 6:
            oneKm = oneKm + 1
        if Dis < 11:
            twoKm = twoKm+1
    onekmpct = (100 * oneKm / (length+0.001))
    twokmpct =(100 * twoKm / (length+0.001))
    print(T,length, "<1KM: ",onekmpct, "<2KM: ",twokmpct)
    onekmlist.append(onekmpct)
    twokmlist.append(twokmpct)
print(len(Topic_stats.index))
print(len(onekmlist))
Topic_stats["1KM"] = onekmlist
Topic_stats["2KM"] = twokmlist

sortedTopics = Topic_stats.sort_values(by= "MSD")
print(sortedTopics.head(30))