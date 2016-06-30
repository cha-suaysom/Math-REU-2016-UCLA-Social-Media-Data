# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:26:20 2016
@author: andy
"""

import pickle
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import pandas as pd
training_data = pickle.load(open('Location_pandas_data_barc.pkl', 'rb'))
#%%
## Generate GMMS
## IMPORTANT!! I multiplied lats and longs by 100 to avoid numerical inaccuracies!
NT = 100#number of topics
n_components_range = np.arange(1,11,1) # run through the number of components you want
gmms =[]
for T in range(0,NT):
    t1 = training_data[training_data["topics"] == T]
    a = t1["latitude"]
    b = t1["longitude"]
    K =len(a)
    lats = np.array(a)
    longs = np.array(b)
    locs = np.column_stack((longs*100,lats*100))
    #aic = [] # you can use either bic or aic. bic penalties n_components more than aic does
    lowest_bic = np.infty
    for n_components in n_components_range:
        print("topic:" + str(T) + " components:" + str(n_components))

        g = mixture.GMM(n_components, 'full')
        g.fit(locs)
        bic_now = g.bic(locs)
        #aic.append(aic_now)
        if bic_now<lowest_bic:
            lowest_bic = bic_now
            best_gmm = g
    gmms.append(best_gmm)
    pickle.dump(best_gmm, open('gmms/gmm_barc_for_topic_'+str(T)+'.pkl','wb'))
#%%
#if you want to load gmms



#%%
## Generate the scores for topics using L_half norm or MSD
maxlat = training_data["latitude"].max()
minlat = training_data["latitude"].min()
maxlong = training_data["longitude"].max()
minlong = training_data["longitude"].min()
print(maxlat, minlat)
print(maxlong, minlong)

Topic_stats = pickle.load(open('topic_stats_pandas.pkl', 'rb'))
msd_array = np.asarray(((Topic_stats["MSD"]).tolist()))



#Generate the score
score = np.exp((-1.5)*msd_array)
print(score)
plt.hist(score)

pickle.dump(score,open('score_of_topics.pkl','wb'))
#%%
