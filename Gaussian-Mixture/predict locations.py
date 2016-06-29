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
training_data = pickle.load(open('training_data.pkl','rb'))
test_data = pickle.load(open('test_data.pkl','rb'))
#%%
gmms = []
for T in range(0,100):
    g = pickle.load(open('gmms/gmm_for_topic_' + str(T) +'.pkl','rb'))
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re  # regex

N = len(test_data)
test_data_text = test_data['text']
clean_text = [" ".join([   # joins a list of words back together with spaces in between them
                                re.sub(r'\W+', '', # force alphanumeric (after doing @ and # checks)
                                word.replace('"','').lower()) # force lower case, remove double quotes
                            for word in tweet.split() # go word by word and keep them if...
                                if len(word)>2 and # they are 3 characters or longer
                                not word.startswith('@') and # they don't start with @, #, or http
                                not word.startswith('#') and
                                not word.startswith('http')]
                            )#.encode('ascii', errors='ignore') # force ascii encoding, ignore weird characters just in case
                        for tweet in test_data_text]
stop_words = [] # stop words file includes English, Spanish, and Catalan
with open('stop_words.txt','r',encoding="utf-8") as f:
    #u = pickle._Unpickler(f)
    #u.encoding = 'latin'
    stop_words = [word.replace("\n",'') for word in f.readlines()] # Have to remove \n's because I didn't copy the stop words cleanly

print("Stop word examples:", stop_words[:10])
print("\n----20 TWEETS----")
for tweet in clean_text[:20]: 
    print(tweet) 
print("--------------")

with open('TF_IDF_feature_names.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin'
    names = u.load()
#tf_idf = TfidfVectorizer(min_df=10,stop_words=stop_words, sublinear_tf= True, vocabulary = names)
#text_tf_idf = tf_idf.fit_transform(clean_text) 

countvectorizer = CountVectorizer(stop_words = stop_words,vocabulary = names)
text_count = countvectorizer.fit_transform(clean_text)

H = pickle.load(open('matrix_H.pkl','rb')) #load the term-topic matrix
H_normalized = H/np.sum(H,axis=0).reshape(1,H.shape[1])
#%%
topic_distributions = text_count * H_normalized.T
## normalize each row so that each row add up to 1.
topic_distributions = topic_distributions / np.sum(topic_distributions,axis=1).reshape(N,1)

#%%
import math
bounds = [(minlong*100,maxlong*100),(minlat*100,maxlat*100)]
predicted_locs = []
Lx = 180 #number of
Ly = 100 #grids
x = np.linspace(minlong*100,maxlong*100,Lx)
y = np.linspace(minlat*100,maxlat*100,Ly)
X,Y = np.meshgrid(x,y)
XX = np.array([X.ravel(),Y.ravel()]).T
Zs = []
topics = []
bounds = [(minlong*100,maxlong*100),(minlat*100,maxlat*100)]

for t in np.arange(0,len(test_data)):
    #print(t)
    topic_distribution = topic_distributions[t]
    # Some of the tweet text consist characters that are not recognizable by ascii, and therefore its topic-distribution is (0,0,0...,0). We want to get rid of that
    if math.isnan(sum(topic_distribution)) == False and max(topic_distribution)>0.0: # You can change 0.0 to any number in [0,1)
        
        Z = np.zeros(Lx*Ly,)
        for i in np.arange(1,100):
            if topic_distribution[i]>0.2:
                Z += score[i]*topic_distribution[i]*np.exp(gmms[i].score(XX))
                #Z += topic_distribution[i]*np.exp(gmms[i].score(XX))
        Z_reshaped = Z.reshape(X.shape)
        result = XX[Z.argmax()]
        result = np.asarray([result[1]/100,result[0]/100])
        print(t,' ', np.argmax(topic_distribution), ' ', result)
        predicted_locs.append(result)
        Zs.append(Z_reshaped)
        # Below is the method that can detect the peak of a continuous function. Running very slow.
        '''
        print(t)
        def func2d(u):
            Z = 0
            for i in np.arange(0,100):
                if topic_distribution[i]>0.05:
                    Z += -score[i]*topic_distribution[i]*math.exp(gmms[i].score(u.reshape(1,-1)))
            return Z
        result = differential_evolution(func2d, bounds)
        result = result.x
        predicted_locs.append(result)
       '''
    else:
        result = 'rejected'
        predicted_locs.append(result)
        Zs.append(np.zeros(X.shape)) #Just give those nonsense tweets (rejected tweets) a nonsense distribution (all zeros)

#print(result.x[1]/100, result.x[0]/100)


#%%
for t in np.arange(0,len(Zs)):
    sumofZ = Zs[t].sum()
    if ((sumofZ==0) == False):
        Zs[t] = Zs[t]/sumofZ

Frac_L = []
unif_Z = np.ones(Lx*Ly)
unif_Z = unif_Z / (Lx*Ly)
L_P_unif = ((np.sqrt(unif_Z)).sum())**2
for Z in Zs:
    if(Z.sum() == 0):
        Frac_L.append(L_P_unif) #Just give those nonsense tweets (rejected tweets) a nonsense Frac_L norm which is really large
    else:
        L_P = ((np.sqrt(Z)).sum())**2
        Frac_L.append(L_P) 

plt.hist(Frac_L, bins=100, range = [0,4000])
#%%
from geopy.distance import vincenty
distances = []
for t in np.arange(0,len(predicted_locs)):
    if predicted_locs[t] != 'rejected' and  Frac_L[t] < 3000: #3000 is supposed to change according to your histogram of Frac_L
        temp = test_data.iloc[t]
        #print(t,' ', temp['latitude'],temp['longitude'], 'vs', predicted_locs[t])
        actual = (temp['latitude'],temp['longitude'])
        distance = vincenty(actual,predicted_locs[t]).km
        distances.append(distance)

print('avg error distance: ',sum(distances)/len(distances)) #avg error distance
print('length of distances list', len(distances))
#%%
#calculate the percentage of predictions that have less than 1km error
count = 0
radius = 1
for t in np.arange(0,len(distances)):
    if distances[t] < radius: 
        count += 1
print(count/len(distances),'% of the predicted tweets have error less than', radius, 'km.')
#%%
print('# of tweets that have lable \"rejected\":', predicted_locs.count('rejected'))
