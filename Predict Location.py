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

Topics = normalized_H*(rest_of_tweets_TFIDF.T)
Topic_list = (np.argmax(Topics.T, axis = 1)).tolist()
print(len(Topic_list))
print(len(rest_of_tweets_Data.index))
rest_of_tweets_Data["topics"]= Topic_list
pickle.dump(rest_of_tweets_Data, open('rest_of_tweets_pandas_data_barc.pkl','wb'))
Topic_stats = pickle.load(open('topic_stats_pandas.pkl', 'rb'))

for T in range(21,22):
    subset= rest_of_tweets_Data[rest_of_tweets_Data["topics"]== T]
    xgrid = subset["xgrid"].tolist()
    ygrid = subset["ygrid"].tolist()
    A = Topic_stats["peak"]
    xgrid =xgrid
    ygrid = ygrid
    length = len(xgrid)
    distance = 0
    for i in range(0,length):
        distance = distance + math.sqrt((xgrid[i]-A[0][0])**2 + (ygrid[i] -A[0][1])**2)

    print(distance/length)
