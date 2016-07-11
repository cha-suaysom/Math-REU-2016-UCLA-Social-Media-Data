import pickle
import numpy as np
import gmplot
import pandas as pd

# pip install gmplot
# use "2to3 -w *.py" to convert gmplot to python 3 (sudo it on unix-likes)
# you have to do this in the directory where the module gets installed
# on my system it goes to /Users/davidnola/anaconda/lib/python3.5/site-packages/gmplot

#df = pickle.load(open('Location_pandas_data_barc.pkl','rb'))
df = pickle.load(open('rest_of_tweets_pandas_data_barc.pkl','rb'))
N = 1
# t1 = df[df['topic']==15]
t1 = df[df['topics']==N]
t1 = t1[t1['gps_precision']==10.0]

print(t1.head())
#gmap = gmplot.GoogleMapPlotter(49.2827, -123.1207, 11)
gmap =gmplot.GoogleMapPlotter(41.390205, 2.154007, 11)

lats = t1['latitude'].tolist()
longs = t1['longitude'].tolist()
print(len(lats))
x1 = ((41.5319-41.2638)*0.40+41.2638)
x2= ((41.5319-41.2638)*0.50+41.2638)
y1= ((2.36254-1.97775)*0.50+1.97775)
y2= ((2.36254-1.97775)*0.60+1.97775)
xx1 = ((41.5319-41.2638)*0.4+41.2638)
xx2= ((41.5319-41.2638)*0.45+41.2638)
yy1= ((2.36254-1.97775)*0.35+1.97775)
yy2= ((2.36254-1.97775)*0.4+1.97775)
squarelats = [x1,x1,x2,x2]
squarelongs = [y1,y2,y2,y1]
squarelats1 = [xx1,xx1,xx2,xx2]
squarelongs1 = [yy1,yy2,yy2,yy1]
gmap.heatmap(lats, longs,dissipating=True,radius=40)
mymap = "mymap_barc"+ str(N)+".html"
gmap.polygon(squarelats, squarelongs ,color = 'blue')
#gmap.polygon(squarelats1, squarelongs1 ,color = 'green')
gmap.draw(mymap)

#print(topics[:10])