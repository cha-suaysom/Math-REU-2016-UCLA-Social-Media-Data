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
gmm_scores = []
Lx = 360 #number of
Ly = 200 #grids
x = np.linspace(minlong*100,maxlong*100,Lx)
y = np.linspace(minlat*100,maxlat*100,Ly)
X,Y = np.meshgrid(x,y)
XX = np.array([X.ravel(),Y.ravel()]).T
for i in np.arange(0,100):
    P = np.exp(gmms[i].score(XX))
    P = P/P.sum() #normalize it?
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

for t in np.arange(0,len(test_data)):
    topic_distribution = topic_distributions[t]
    greatest_topic = np.argmax(topic_distribution)
    # Some of the tweet text consist characters that are not recognizable by ascii, and therefore its topic-distribution is (0,0,0...,0). We want to get rid of that
    if math.isnan(sum(topic_distribution)) == False and greatest_topic != 0 and max(topic_distribution)>0.3: # You can change 0.0 to any number in [0,1)
        
        Z = np.zeros(Lx*Ly,)
        topic_distr_sum = 0
        for i in np.arange(1,100):
            if topic_distribution[i]>0.1:
                topic_distr_sum += topic_distribution[i]
                Z += score[i]*topic_distribution[i]*gmm_scores[i]
        Z_reshaped = Z.reshape(X.shape)
        sumofZ = Z_reshaped.sum()
        if sumofZ == 0:
            result = 'rejected'
            Frac_L.append(L_P_unif)
        else:
            result = XX[Z.argmax()]
            result = np.asarray([result[1]/100,result[0]/100])
            Z_reshaped = Z_reshaped/sumofZ
            L_P = ((np.sqrt(Z_reshaped)).sum())**2
            Frac_L.append(L_P) 
            print(t,' ', greatest_topic, ' ', result)
        predicted_locs.append(result)

    else:
        result = 'rejected'
        predicted_locs.append(result)
        Frac_L.append(L_P_unif)
    topics.append(greatest_topic)

percent_rejected = predicted_locs.count('rejected')/len(predicted_locs) * 100
print(predicted_locs.count('rejected'), 'out of', len(predicted_locs),'(',percent_rejected,'%)' 'tweets have been rejected to given a predicted locs.')
print(len(predicted_locs) - predicted_locs.count('rejected'), 'out of', len(predicted_locs),'(',100-percent_rejected,'%)', 'tweets have been given a predicted locs.')

#%%
from geopy.distance import vincenty
distances = []
#lat_to_km = 111
#long_to_km = 111*math.cos(49/180*math.pi)
threshold_Frac_L = np.percentile(Frac_L,5)
for t in np.arange(0,len(predicted_locs)):
    if predicted_locs[t] != 'rejected' and  Frac_L[t] < threshold_Frac_L: 
        #print(t)        
        temp = test_data.iloc[t]
        #print(t,' ', temp['latitude'],temp['longitude'], 'vs', predicted_locs[t])
        actual = (temp['latitude'],temp['longitude'])
        #distance = ((lat_to_km*(actual[0]-predicted_locs[t][0]))**2 + ((long_to_km*(actual[1]-predicted_locs[t][1]))**2))**0.5
        distance = vincenty(actual,predicted_locs[t]).km
        distances.append(distance)
    else:
        True
        #distances.append('NA')
print('avg error distance: ',sum(distances)/len(distances)) #avg error distance
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
df = pd.DataFrame(columns = ( "topic", "predicted_loc","actual_loc","Frac_L", "topic_by_W"))
for t in np.arange(0,len(topics)):
    temp = test_data.iloc[t]
    actual = (temp['latitude'],temp['longitude'])
    actual_locs.append(actual)
df['topic'] = topics
df['predicted_loc'] = predicted_locs
df['actual_loc'] = actual_locs
df['Frac_L'] = Frac_L
topic_by_W = test_data[0:len(topics)]['topic'].tolist()
df['topic_by_W'] = topic_by_W
pickle.dump(df,open('predict_results','wb'))
