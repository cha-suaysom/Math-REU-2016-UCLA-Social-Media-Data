# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:40:55 2016

@author: andy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:33:58 2016

@author: andy
"""

import pickle
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import calendar
abrev_num = list(calendar.day_abbr), range(7)

with open('NMF_100_topics_vanc_WH.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin'
    (W, H) = u.load()
#(W, H) = pickle.load(open('NMF_100_topics_vanc_WH.pkl','rb'))
print(W.shape)
print(H.shape)
#%%
with open('TF_IDF_feature_names.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin'
    names = u.load()
#names = np.array(pickle.load(open('TF_IDF_feature_names.pkl','rb')))
#%%
NT = 100 #number of topics
with open('pandas_data_vanc.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin'
    raw_data = u.load()
#raw_data = pickle.load(open('pandas_data_vanc.pkl','rb'))

Topics = W.argmax(axis=1)#Assigns a topic to each tweet
raw_data["topic"] = Topics#data frame containing the valuable columns from raw data
Time = raw_data["time"]
raw_data["date"] = pd.to_datetime(Time, infer_datetime_format= True)
#%%
start = dt.date(2012,1,1)
end = dt.date(2014,12,31)
hours = np.arange(0,24)
N=24

TemporalList= []
#%% T stands for topic index, since there are 100 topics, it can range from 0 to 99
# if T = 100, calculate all topics
def calculate_time_distribution(T):
    if T==100:
        General = np.zeros(N)
        TopicDf = raw_data
        datelist = TopicDf["date"].tolist()
        K= len(datelist)
        General= np.zeros(N)

        for i in range(0,K):
            t = datelist[i].hour
            General[t] = General[t] + 1
        General= General/K
        #plt.bar(hours,General)
        return General
    elif T<0 or T>100:
        return 0
    else:
        TopicDf= raw_data[raw_data["topic"]== T]
        datelist = TopicDf["date"].tolist()
        K= len(datelist)
        Topic= np.zeros(N)
        
        for i in range(0,K):
            t = datelist[i].hour
            Topic[t] = Topic[t] + 1
        Topic= Topic/K
        #plt.bar(hours,Topic)
        return Topic
#%%
for T in np.arange(0,100):
        TemporalList.append(calculate_time_distribution(T))


#%%
General = calculate_time_distribution(100)

#%%
from pyemd import emd
distance_matrix = np.zeros(shape=(24,24))
for i in np.arange(0,24):
    for j in np.arange(0,24):
        distance_matrix[i,j] = min(abs(i-j),24-abs(i-j))
        
#%%
def calculate_distance_from_general(T):
    if T>=0 and T<100:
        Topic_T = TemporalList[T]
        #dist_T = np.linalg.norm(Topic_T-General) #this is the Euclidian norm   
        dist_T = emd(General,Topic_T,distance_matrix)
        return dist_T

Temporal_diff =[]
for T in np.arange(0,100):
    dist = calculate_distance_from_general(T)
    Temporal_diff.append(dist)
Temporal_diff = np.asarray(Temporal_diff)
rank_hour = Temporal_diff.argsort()

