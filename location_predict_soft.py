import pandas as pd
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import *


p = 0.7 #percentage of training data

# loading data 
(W, H) = pickle.load(open('NMF_100_topics_WH.pkl', 'rb'))
names = np.array(pickle.load(open('TF_IDF_feature_names.pkl', 'rb')))
print(W.shape)
print(H.shape)
NT = 100  #number of topics
Nt = W.shape[0]  #number of tweetsR

# load training data and test data and combine them (Andy's code exports training and test data separately)
#From Location Matrix.py Location_pandas_data_barc.pkl -> training_data Rest_of....->Test_data
training_data = pickle.load(open('training_data.pkl', 'rb'))
test_data = pickle.load(open('test_data.pkl', 'rb'))
list1 = [training_data, test_data]
Spatial = pd.concat(list1)

# Normalize W so that we can interpret it as prob distribution
row_sums = W.sum(axis=1)
W_normal = W / row_sums[:, np.newaxis]
where_are_NaNs = isnan(W_normal)
W_normal[where_are_NaNs] = 0
print(W_normal.shape)


# find the range of the latitude and longitude
maxlat = Spatial["latitude"].max()
minlat = Spatial["latitude"].min()
maxlong = Spatial["longitude"].max()
minlong = Spatial["longitude"].min()
print(maxlat, minlat)
print(maxlong, minlong)


# I used grids of size .01 deg by .01
#hx = 1/(111*math.cos(maxlat * math.pi / 180))
#hy = 1/111
#hx = 0.01
#hy = 0.01
#Nx = math.ceil((maxlong - minlong) / hx)
#Ny = math.ceil((maxlat - minlat) / hy)

# size of grids
Nx = 100  
Ny = 180
hx = (maxlong - minlong) / Nx
hy = (maxlong - minlong) / Ny
print(Nx)
print(Ny)


# Spatial_precise = Spatial[Spatial["gps_precision"] == 10.0]#taking gonly location accurete tweets
#
# distr = []
# for T in range(0, NT):
#     pdf = np.zeros((Nx, Ny))
#     distr.append(pdf)
# for t in range(0, Nt):
#     if Spatial.iloc[t]["gps_precision"] == 10:
#         x = Spatial.iloc[t]["longitude"]
#         y = Spatial.iloc[t]["latitude"]
#         x_bin = math.floor((x-minlong)/hx)
#         y_bin = math.floor((y-minlat)/hy)
#         print(x_bin)
#         print(y_bin)
#         for T in range(0, NT):
#             distr[T][x_bin, y_bin] += W[t, T]
#             #pdf[math.floor((x-minlat)/hx), math.floor((y-minlong)/hy)] += W[t, T]
#     print(t)


#BUILD PHASE
# This is where we calculate the distribution for each topic
# change "build" to 1 if we want to create the distribution for topics
build = 0
if build:
    data = []

    # T for Topic, t for tweet
    for T in range(0, NT):
        pdf = np.zeros((Nx, Ny))
        for t in range(0, math.floor(Nt*p)):

            # find where the tweet occured
            x = Spatial.iloc[t]["longitude"]
            y = Spatial.iloc[t]["latitude"]
            x_bin = math.floor((x-minlong)/hx)
            y_bin = math.floor((y-minlat)/hy)
            pdf[x_bin, y_bin] += W[t, T]  # add the weight on topic T of tweet t to the correct bin
        sums = pdf.sum()
        pdf_normal = pdf / sums # normalize to a probability distribution
        name = 'distr' + str(T) + '.dat'
        pdf_normal.dump(name) # save the distribution so we don't have to calculate it every time
        data.append(pdf_normal)
        print(T)

# load data from file if we are not building the distribution from scratch
else:
    data = []
    for i in range(0, NT):
        name = 'distr' + str(i) + '.dat'
        test = np.load(name)
        data.append(test)




# turn "test" to 1 if we want to test the model
test = 1
# LPlist = []
# for t in range(0, Nt):
#     distribution = np.zeros((Nx, Ny))
#     for T in range(0, NT):
#         distribution += (W_normal[t, T] * data[T]) # Increment the distribution function in the correct grid according to the weight
#
#     L_P = ((np.sqrt(distribution)).sum()) ** 2
#     LPlist.append(L_P)  # Add the L^0.5 norm of the distribution to the list
#     print(t)
# pickle.dump(LPlist, open("LPlist.p", "wb"))
LPlist = pickle.load(open('LPlist.p', 'rb'))
LPlist_nonzero = [x for x in LPlist if x != 0]

LP_threshold = np.percentile(LPlist_nonzero, 20)
print(LP_threshold)
# TEST PHASE
# Ignore this part of the code if you are using a separate test file
if test:
    count = 0
    error = 0
    p = 0.7  # the percentage of data for training
    predict = [[], [], []]
    exact = 0
    onekm = 0
    twokm = 0
    for t in range(math.ceil(Nt * p), Nt):
        if LPlist[t] < LP_threshold:
            # calculate the location of the tweet
            x = Spatial.iloc[t]["longitude"]
            y = Spatial.iloc[t]["latitude"]
            x_bin = math.floor((x - minlong) / hx)
            y_bin = math.floor((y - minlat) / hy)

            distribution = np.zeros((Nx, Ny))
            for T in range(0, NT):
                distribution += (W_normal[t, T] * data[T])  # Calculate the distribution of the tweet
            # find the bin with highest density function
            if distribution.sum() > 0.1:
                count += 1
                indices = np.where(distribution == distribution.max())
                x_bin_predict = indices[0][0]
                y_bin_predict = indices[1][0]

                # record the predictions and the tweet id in the variable "predict"
                predict[0].append(Spatial.iloc[t]["tweet_id"])
                predict[1].append(x_bin_predict)
                predict[2].append(y_bin_predict)

                #print(np.absolute(x_bin - x_bin_predict))
                #print(np.absolute(y_bin - y_bin_predict))

                # calculate the distance from the correct bin
                diffX = np.absolute(x_bin - x_bin_predict) * 230
                diffY = np.absolute(y_bin - y_bin_predict) * 250
                diff = math.sqrt(diffX ** 2 + diffY ** 2)
                error += diff

                if diff < 2000:
                    twokm += 1
                    if diff < 1000:
                        onekm += 1
                        if diff == 0:
                            exact += 1

    print(count, Nt*(1-p))  # the average error

    print("percentage for exact is:")
    print(exact / count)

    print("percentage for 1km is:")
    print(onekm / count)

    print("percentage for 2km is:")
    print(twokm / count)
    
