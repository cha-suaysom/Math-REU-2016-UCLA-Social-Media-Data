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
raw_data = pickle.load(open('pandas_data_vanc_withtopics.pkl','rb'))
Spatial = raw_data[raw_data['gps_precision'] == 10]
#%%
shuffled_data = Spatial.sample(frac=1)
#%%
Nt = len(shuffled_data) #1988182
N_training = int(Nt*0.7) #1391727
#training_data = shuffled_data[0:N_training]
#test_data = shuffled_data[N_training:]
#%% save it so we do not need to generate it again
#pickle.dump(shuffled_data,open('shuffled_data.pkl','wb'))
#pickle.dump(test_data,open('test_data.pkl','wb'))
#%% Next time, we only need to load these files
shuffled_data = pickle.load(open('shuffled_data.pkl','rb'))
training_data = pickle.load(open('training_data.pkl','rb'))
test_data = pickle.load(open('test_data.pkl','rb'))
#%%
## Generate GMMS
## IMPORTANT!! I multiplied lats and longs by 100 to avoid numerical inaccuracies!
NT = 100
n_components_range = np.arange(1,11,1) # run through the number of components you want
gmms =[]
for T in range(0,100):
    t1 = training_data[training_data["topic"] == T]
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
    pickle.dump(best_gmm, open('gmms/gmm_for_topic_'+str(T)+'.pkl','wb'))
#%%
#if you want to load gmms

gmms = []
for T in range(0,100):
    g = pickle.load(open('gmms_most_110/gmms/gmm_for_topic_'+str(T)+'.pkl','rb'))
    gmms.append(g)

#%%
## Generate the scores for topics using L_half norm or MSD
maxlat = training_data["latitude"].max()
minlat = training_data["latitude"].min()
maxlong = training_data["longitude"].max()
minlong = training_data["longitude"].min()
print(maxlat, minlat)
print(maxlong, minlong)
MSD_List = []#stores the Mean Square Distance between all the tweets in  agiven topic
Topics_Size = []
for T in range(0,NT):#Mean Square Distance Calculation: we want to find the sum of the square of the
    x = (training_data[training_data["topic"] == T])# euclidean pairwise distances between all the tweets in each topic divided by the size of the topic (K)
    a = x["latitude"]
    b = x["longitude"]
    K =len(a)
    Topics_Size.append(K)
    X = np.array(a)#we calculate x axis pairwise square distances
    Y = np.array(b)#and then seperatley y axis distances
    MSD = 1.0*(((K+1)*np.dot(X.T, X) - (X.sum())**2)+((K+1)*np.dot(Y.T, Y) - (Y.sum())**2))/((K+1)**2)#I am almost 100% this is correct but tell me if you guys see anything alarming
    MSD_List.append(MSD)#in the above, we used K+1 instead of K (Where K is the size of the topic) in the denomenator to make  sure we ar enot dividing by 0 for the empty topics
df = pd.DataFrame(columns = ( "Length", "MSD"))#save the topics size and MSD into a data frame
df["MSD"] = MSD_List
df["Length"] = Topics_Size

#next we want to create a "density function" of the tweets for each topic over the city of barcelona in order to calculate other location statistics
#to do this we partition our location grid of [0,1]x[0,1] into LxL squares (here L = 100) we create matrices for each of the 500 topics to store the
#the number of tweets in each of the squares, we later divide the matrix by the total number of tweets in the topic to get a probibistic matrix of
# where are the tweets distrbuted for each topic

Lx = 27 #since the "length" of Vancouver is about 1.8 times the "width" of it,
Ly = 15 # I divide Vancouver into a 27*14 grid. If your computer is powerful, you can make the numbers bigger. However, emd has a time complexity of O(n^3) where n is the number of grids.
ArrayList = []
for T in range(0,NT):#F density calculation
    A = np.zeros((Ly,Lx))
    G = training_data[training_data["topic"] == T]
    G["latitude"] = (G["latitude"]-minlat)/(maxlat-minlat)#we normalize the latatitudal and longitudal data of the tweets to be between 0 and 1
    G["longitude"] = (G["longitude"]-minlong)/(maxlong-minlong)
    Glong = G["longitude"].tolist()
    Glat = G["latitude"].tolist()    
    N = len(Glat)
    for i in range(0,N):
        x = int((Glong[i]*(Lx-1*10**(-12))))#subtracted 10^-12 from L which is neglible but needed to make sure that the one tweet with the maximal longitude
        y = int((Glat[i]*(Ly-1*10**(-12))))#(which is equal to 1 once normalized) was not causing an out of bounds error when used as an index for the array
        A[y,x] = A[y,x]+ 1
    ArrayList.append(A/N)

MetricList= []
for X in ArrayList:
    L_P = ((np.sqrt(X)).sum()*(1.0/(Lx*Ly)))**2
    L_1 = (X.sum()+1*10**(-12))*(1.0/(Lx*Ly))
    Final_L = L_P/L_1# L^0.5 norm divided by L^1 norm
    MetricList.append(Final_L)
df["L0.5"]= MetricList

# we can see the ranking of each topic
msd_array = np.asarray(MSD_List)
print('Ranking of MSD')
msd_array.argsort()

metric_array = np.asarray(MetricList)
print('Ranking of Lhalf')
metric_array.argsort()
#%%
#Generate the score
score = np.exp((-1.5)*msd_array)
#score = np.exp((-5)*metric_array)
#score = np.power(msd_array,-0.8)
print(score)
plt.hist(score)

pickle.dump(score,open('score_of_topics.pkl','wb'))
#%%
# Draw the contour map in the console, 18th topic is the yvr topic
Topic_N  = 39
#t1 = raw_data[raw_data["topic"] == Topic_N]
t1 = training_data[training_data['topic'] == Topic_N]
a = t1["latitude"]
b = t1["longitude"]
K =len(a)
lats = np.array(a)
longs = np.array(b)
locs = np.column_stack((longs,lats))
maxlat = max(lats)
maxlong = max(longs)
minlat = min(lats)
minlong = min(longs)
g = gmms[Topic_N]
x = np.linspace(minlong,maxlong,90)
y = np.linspace(minlat,maxlat,50)
X,Y = np.meshgrid(x,y)
XX = np.array([X.ravel(),Y.ravel()]).T
Z = g.score(XX*100) # Z is the log of probability of the 90*50 sample points
minZ = min(Z)
maxZ = max(Z)
Z = Z.reshape(X.shape)
CS = plt.contour(X,Y,Z,levels = np.linspace(minZ,maxZ,10))
CB = plt.colorbar(CS,shrink=0.8, extend = 'both')
plt.scatter(locs[:,0], locs[:,1],.1)
plt.show()
#%%
# Draw the contour map on google map (generate .kml files, which can be opened using My Maps on Google)
import matplotlib.cm
import matplotlib.colors
import numpy as np

def output_kml_segment(points, level, o):
  o.write('<Placemark>\n')
  o.write('<name>%d</name>\n'%(level))
  o.write('<styleUrl>#%d</styleUrl>\n'%(level))
  o.write("<LineString>\n")
  o.write("<coordinates>\n")
  for p in points:
    o.write("%.6f,%.6f "%(p[0],p[1]))
  o.write("\n</coordinates>\n")
  o.write("</LineString>\n")
  o.write('</Placemark>\n')

def output_kml_segments(segs, level, o):
  for s in segs:
    output_kml_segment(s,level,o)  

def output_kml_levels(levels, o):
  ## Our own colormap
  cmap = matplotlib.colors.LinearSegmentedColormap.from_list(colors=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)), name='mybgr')
  matplotlib.cm.register_cmap(cmap=cmap)
  scale = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(levels[0],levels[-1]), plt.get_cmap("mybgr"))
  for l in levels:
    o.write('<Style id="%d">\n'%(l))
    o.write('<LineStyle>\n')
    o.write('<width>2</width>\n')
    o.write('<gx:labelVisibility>1</gx:labelVisibility>\n')
    c = scale.to_rgba(l)
    o.write('<color>ff%02x%02x%02x</color>\n'%(int(c[2]*255),int(c[1]*255),int(c[0]*255)))
    #<gx:labelVisibility>1</gx:labelVisibility>
    o.write('</LineStyle>\n')
    o.write('</Style>\n')
def output_kml(cs,o):
  o.write('<?xml version="1.0" encoding="UTF-8"?>\n')
  o.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
  o.write('<Document>\n')
  output_kml_levels(cs.levels,o)
  for i in range(0,len(cs.levels)):
    #print "level ",cs.levels[i]," has ",len(cs.allsegs[i])," segments"
    output_kml_segments(cs.allsegs[i], cs.levels[i], o)
  o.write('</Document>\n')
  o.write('</kml>')
output_kml(CS,open(str(Topic_N) + ".kml", "w+"))
#%%
# Find the peak of the pdf using differential_evolution method
from scipy.optimize import differential_evolution
bounds = [(minlong*100,maxlong*100),(minlat*100,maxlat*100)]
def func2d(x):
    return -np.exp(g.score(x.reshape(1,-1)))
result = differential_evolution(func2d, bounds)
print(result.x[1]/100, result.x[0]/100) #the lat and long of the peak