# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:26:46 2016
@author: andy
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#%%
training_data = pickle.load(open('Location_pandas_data_barc.pkl', 'rb'))
test_data = pickle.load(open('rest_of_tweets_pandas_data_barc.pkl','rb'))
N = len(test_data)
#%%
gmms = []
for T in range(0,100):
    g = pickle.load(open('gmms/gmm_barc_for_topic_' + str(T) +'.pkl','rb'))
    gmms.append(g)

#%%
a = training_data['latitude'].tolist()
b = training_data['longitude'].tolist()
minlong = min(b)
maxlong = max(b)
minlat = min(a)
maxlat = max(a)
score = pickle.load(open('score_of_topics.pkl','rb'))
#%%

topic_distributions = pickle.load( open('test_topic_ditribution.pkl', 'rb'))
topic_distributions = topic_distributions.T
## normalize each row so that each row add up to 1.
topic_distributions = topic_distributions / np.sum(topic_distributions,axis=1).reshape(N,1)
#%%
gmm_scores = []
Lx = 100 #number of
Ly = 100 #grids
x = np.linspace(minlong*100,maxlong*100,Lx)
y = np.linspace(minlat*100,maxlat*100,Ly)
X,Y = np.meshgrid(x,y)
XX = np.array([X.ravel(),Y.ravel()]).T
for i in np.arange(0,100):
    P = np.exp(gmms[i].score(XX))
    #P = P/P.sum() #normalize it?
    gmm_scores.append(P)

#%%
import math
predicted_locs = []
actual_locs=[]
topics = []
unif_Z = np.ones(Lx*Ly)
unif_Z = unif_Z / (Lx*Ly)
L_P_unif = ((np.sqrt(unif_Z)).sum())**2
Frac_L = []
#probs = []
for t in np.arange(0,len(test_data)):
    topic_distribution = topic_distributions[t]
    greatest_topic = np.argmax(topic_distribution)
    # Some of the tweet text consist characters that are not recognizable by ascii, and therefore its topic-distribution is (0,0,0...,0). We want to get rid of that
    if math.isnan(sum(topic_distribution)) == False and greatest_topic != 0 and max(topic_distribution)>0.3: # You can change 0.0 to any number in [0,1)
        #temp = test_data.iloc[t]
        #actual = [temp['longitude']*100, temp['latitude']*100]
        #actual = np.asarray(actual).reshape(1,-1)
        Z = np.zeros(Lx*Ly,)
        #norm_coeff = 0
        #prob = 0
        for i in np.arange(1,100):
            if topic_distribution[i]>0.1:
                weight = score[i]*topic_distribution[i]
                Z += weight*gmm_scores[i]
                #norm_coeff += weight
                #prob += weight*np.exp(gmms[i].score(actual))
        Z_reshaped = Z.reshape(X.shape)
        sumofZ = Z_reshaped.sum()
        if sumofZ == 0:
            result = 'rejected'
            Frac_L.append(L_P_unif)
            #probs.append('NA')
        else:
            result = XX[Z.argmax()]
            result = np.asarray([result[1]/100,result[0]/100])
            Z_reshaped = Z_reshaped/sumofZ
            L_P = ((np.sqrt(Z_reshaped)).sum())**2
            Frac_L.append(L_P)
            #prob = prob/norm_coeff
            #probs.append(prob)
            #print(t,' ', greatest_topic, ' ', result)
            print(t)
        predicted_locs.append(result)

    else:
        result = 'rejected'
        predicted_locs.append(result)
        Frac_L.append(L_P_unif)
        #probs.append('NA')
    topics.append(greatest_topic)

percent_rejected = predicted_locs.count('rejected')/len(predicted_locs) * 100
print(predicted_locs.count('rejected'), 'out of', len(predicted_locs),'(',percent_rejected,'%)' 'tweets have been rejected to given a predicted locs.')
print(len(predicted_locs) - predicted_locs.count('rejected'), 'out of', len(predicted_locs),'(',100-percent_rejected,'%)', 'tweets have been given a predicted locs.')

#%%
#from geopy.distance import vincenty
distances = []
lat_to_km = 111
long_to_km = 111*math.cos(41/180*math.pi)
threshold_Frac_L = np.percentile(Frac_L,5)
for t in np.arange(0,len(predicted_locs)):
    if predicted_locs[t] != 'rejected' and  Frac_L[t] < threshold_Frac_L:
        #print(t)
        temp = test_data.iloc[t]
        #print(t,' ', temp['latitude'],temp['longitude'], 'vs', predicted_locs[t])
        actual = (temp['latitude'],temp['longitude'])
        distance = ((lat_to_km*(actual[0]-predicted_locs[t][0]))**2 + ((long_to_km*(actual[1]-predicted_locs[t][1]))**2))**0.5
        #distance = vincenty(actual,predicted_locs[t]).km
        distances.append(distance)
    else:
        True
        #distances.append('NA')
print('avg error distance: ',sum(distances)/len(distances)) #avg error distance
print('median error distance:', np.percentile(distances,50))
print('length of distances list', len(distances))
#%%
#calculate the percentage of predictions that have less than 1km error
count = 0
radius = 1
for t in np.arange(0,len(distances)):
    if distances[t] < radius:
        count += 1
print(count/len(distances)*100,'% of the predicted tweets have error less than', radius, 'km.')
#%%
df = pd.DataFrame(columns = ( "topic", "predicted_loc","actual_loc","Frac_L"))
for t in np.arange(0,len(topics)):
    temp = test_data.iloc[t]
    actual = (temp['latitude'],temp['longitude'])
    actual_locs.append(actual)
df['topic'] = topics
df['predicted_loc'] = predicted_locs
df['actual_loc'] = actual_locs
df['Frac_L'] = Frac_L
pickle.dump(df,open('predict_results.pkl','wb'), protocol = 4)