# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 08:53:00 2016

@author: andy
"""

import pandas as pd
import pickle
import numpy as np
#import matplotlib.pyplot as plt
import gmplot
import calendar
abrev_num = list(calendar.day_abbr), range(7)
#%%
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
#%%
Topics = W.argmax(axis=1)#Assigns a topic to each tweet
raw_data["topic"] = Topics#data frame containing the valuable columns from raw data
Time = raw_data["time"]
raw_data["date"] = pd.to_datetime(Time, infer_datetime_format= True)

Spatial = raw_data[raw_data['gps_precision'] == 10]
#Spatial = Spatial[Spatial["latitude"]>41]#removed one outlier with very low lattitude

#%%
def generate_topic_heatmap(Topic_N, interval_width = 24, user = "defaultValue"):
    t1 = Spatial[Spatial['topic']==Topic_N]
   
    if (user != "defaultValue"):
        t1 = Spatial[Spatial['user'] == user]
    else:
        user = ""
    TemporalDistr = []
    for i in np.arange(0,int(24/interval_width)):
        TemporalDistr.append([])
    datelist = t1["date"].tolist()
    K= len(datelist)
    for i in range(0,K):
         t = datelist[i].hour
         TemporalDistr[int(t/interval_width)].append(t1.iloc[i])
    
    for t in np.arange(0,int(24/interval_width)):
        gmap = gmplot.GoogleMapPlotter(49.2827, -123.1207, 11)
        x = pd.DataFrame(TemporalDistr[t])
        lats = x['latitude'].tolist()
        longs = x['longitude'].tolist()
        print(len(lats))
        gmap.heatmap(lats, longs,dissipating=True,radius=40)
        mymap = "mymap_vanc_"+ str(Topic_N)+user+"_interval_width_"+str(interval_width)+"_hour_"+str(int(t*interval_width))+".html"
        gmap.draw(mymap)
        
#generate_topic_heatmap(43)
#generate_topic_heatmap(43,3)
generate_topic_heatmap(56,3)
#%%
def generate_topic_heatmap_weekdays(Topic_N, user = "defaultValue"):
    t1 = Spatial[Spatial['topic']==Topic_N]

    if (user != "defaultValue"):
        t1 = Spatial[Spatial['user'] == user]
    else:
        user = ""
    TemporalDistr = []
    for i in np.arange(0,7):
        TemporalDistr.append([])
    datelist = t1["date"].tolist()
    K= len(datelist)
    for i in range(0,K):
         t = datelist[i].weekday()
         TemporalDistr[t].append(t1.iloc[i])
    
    
    for t in np.arange(0,7):
        x = pd.DataFrame(TemporalDistr[t])
       
        lats = x['latitude'].tolist()
        longs = x['longitude'].tolist()
        print(len(lats))
        gmap = gmplot.GoogleMapPlotter(49.2827, -123.1207, 11)
        gmap.heatmap(lats, longs,dissipating=True,radius=40)
        mymap = "mymap_vanc_"+  str(Topic_N)+user +abrev_num[0][t]+".html"
        gmap.draw(mymap)
#generate_topic_heatmap_weekdays(35)
#%%
generate_topic_heatmap_weekdays(35,'MikeDangeli')
generate_topic_heatmap_weekdays(35)