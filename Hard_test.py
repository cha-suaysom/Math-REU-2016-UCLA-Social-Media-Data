import pandas as pd
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import *

rows = 100
cols = 100
# load W, H, and the data
(W, H) = pickle.load(open('location_NMF_100_topics_barc_WH.pkl','rb'))
H = H[:,-rows*cols:]
print(W.shape)
print(H.shape)
NT = 100 #number of topics
Nt = W.shape[0] #number of tweets
Spatial = pickle.load(open('pandas_data_vanc.pkl','rb'))




# calculate the min and max for latitude and longitude
maxlat = Spatial["latitude"].max()
minlat = Spatial["latitude"].min()
maxlong = Spatial["longitude"].max()
minlong = Spatial["longitude"].min()
print(maxlat, minlat)
print(maxlong, minlong)

Topic_stats = pickle.load(open('topic_stats_pandas.pkl', 'rb'))
rest_of_tweets_Data = pickle.load(open('rest_of_tweets_pandas_data_barc.pkl','wb'))

for T in range(0,1):
    Location = rest_of_tweets_Data[rest_of_tweets_Data["topics"] == T]
    length = len(Location.index)
    Topic_xy = (Topic_stats[Topic_stats.index == T]["peak"])
    #Topic_y = Topic_xy[1]
    Topic_x = Topic_xy[0]
    print(Topic_xy)
    distance = 0
    for row in Location.itertuples():
        # print(len(row))
        y = ((float(row[7]) - minlat) / (maxlat - minlat)) * rows
        y = math.floor(y)
        x = ((float(row[8]) - minlong) / (maxlong - minlong)) * cols
        x = math.floor(x)
        distance = distance + math.sqrt((x-Topic_x)**2 +(y-Topic_y)**2)
