import pandas as pd
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import *

p = 0.5 # this is determined by Location matrix.py
(W, H) = pickle.load(open('location_NMF_100_topics_barc_WH.pkl', 'rb'))
W_test = pickle.load(open('test_topic_ditribution.pkl', 'rb'))


names = np.array(pickle.load(open('TF_IDF_feature_names.pkl', 'rb')))
print(W.shape)
print(H.shape)


Spatial_training = pickle.load(open('Location_pandas_data_barc.pkl', 'rb'))  # Spatial here already only has location precise data
Spatial_test = pickle.load(open('rest_of_tweets_pandas_data_barc.pkl', 'rb'))  # Spatial here already only has location precise data
#Spatial = pickle.load(open('AllPanda.pkl', 'rb'))  # Spatial here already only has location precise data


print('Training data size = ' + str(len(Spatial_training)))
print('Test data size = ' + str(len(Spatial_test)))
NT = 100  #number of topics
Nt_training = W.shape[0]
Nt_test = len(Spatial_test)
#print('Total data size = ' + str(len(Spatial)))

# # Normalize W so that we can interpret it as prob distribution
row_sums = W.sum(axis=1)
W_normal = W / row_sums[:, np.newaxis]
where_are_NaNs = isnan(W_normal)
W_normal[where_are_NaNs] = 0
print(W_normal.shape)

W_test = W_test.T
row_sums = W_test.sum(axis=1)
W_test_normal = W_test / row_sums[:, np.newaxis]
where_are_NaNs = isnan(W_test_normal)
W_test_normal[where_are_NaNs] = 0
print(W_test_normal.shape)


# find the range of the latitude and longitude
maxlat = Spatial_training["latitude"].max()
minlat = Spatial_training["latitude"].min()
maxlong = Spatial_training["longitude"].max()
minlong = Spatial_training["longitude"].min()
print(maxlat, minlat)
print(maxlong, minlong)
#
#
# # grids of size .01 deg by .01
# #hx = 1/(111*math.cos(maxlat * math.pi / 180))
# #hy = 1/111
# hx = 0.01
# hy = 0.01
# Nx = math.ceil((maxlong - minlong) / hx)
# Ny = math.ceil((maxlat - minlat) / hy)
# print(Nx)
# print(Ny)
#

# Assign number of grids
Nx = 100
Ny = 100
hx = (maxlong - minlong) / Nx
hy = (maxlong - minlong) / Ny


#BUILD PHASE
# This is where we calculate the distribution for each topic
build = 0

if build:

    data = []
    for T in range(0, NT):
        data.append(np.zeros((Nx, Ny)))
    for t in range(0, Nt_training):
        x = Spatial_training.iloc[t]["longitude"]
        y = Spatial_training.iloc[t]["latitude"]
        x_bin = math.floor((x - minlong) / hx)
        y_bin = math.floor((y - minlat) / hy)
        if x_bin == 100:
            x_bin = 99
        if y_bin == 100:
            y_bin = 99
        print(t)
        for T in range(0, NT):
            data[T][x_bin, y_bin] += W[t, T]

    for T in range(0, NT):
        pdf = data[T]
        sums = pdf.sum()
        pdf_normal = pdf / sums  # normalize to a probability distribution
        name = 'distribution_with_location_barc' + str(T) + '.dat'
        pdf_normal.dump(name)  # save the distribution so we don't have to calculate it every time
        data.append(pdf_normal)
        print(T)


# if build:
#     data = []
#     for T in range(0, NT):
#         pdf = np.zeros((Nx, Ny))
#         for t in range(0, Nt_training):
#             x = Spatial_training.iloc[t]["longitude"]
#             y = Spatial_training.iloc[t]["latitude"]
#             x_bin = math.floor((x-minlong)/hx)
#             y_bin = math.floor((y-minlat)/hy)
#             pdf[x_bin, y_bin] += W[t, T]  # add the weight on topic T of tweet t to the correct bin
#         sums = pdf.sum()
#         pdf_normal = pdf / sums # normalize to a probability distribution
#         name = 'distribution_with_location_barc' + str(T) + '.dat'
#         pdf_normal.dump(name) # save the distribution so we don't have to calculate it every time
#         data.append(pdf_normal)
#         print(T)
#
# load data from file if we are not building the distribution from scratch
else:
    data = []
    for i in range(0, NT):
        name = 'distribution_with_location_barc' + str(i) + '.dat'
        test = np.load(name)
        data.append(test)
#
# knob for constructing LP list
constructLP = 0

if constructLP:
    LPlist = []
    for t in range(0, Nt_test):
        distribution = np.zeros((Nx, Ny))
        for T in range(0, NT):
            distribution += (W_test_normal[t, T] * data[T]) # Increment the distribution function in the correct grid according to the weight

        L_P = ((np.sqrt(distribution)).sum()) ** 2
        LPlist.append(L_P)  # Add the L^0.5 norm of the distribution to the list
        print(t)
    pickle.dump(LPlist, open("Location_barc_LPlist.p", "wb"))
else:
    LPlist = pickle.load(open('Location_barc_LPlist.p', 'rb'))


constructL2norm = 0

if constructL2norm:
    L2list = []
    for t in range(0, Nt_test):
        norm = np.linalg.norm(W_test[t])
        L2list.append(norm)
        print(t)
    pickle.dump(L2list, open("Location_barc_L2list.p", 'wb'))
else:
    L2list = pickle.load(open('Location_barc_L2list.p', 'rb'))


## knob for testing
test = 1




output = []
for i in range(1, 61, 10):
    for j in range(20, 91, 10):

        error = 0
        # predict = [[], [], []]
        exact = 0
        onekm = 0
        twokm = 0
        count = 0

        LPtestPercentage = i   # find the 10% tweets with lowest Lp norm
        LPlist_nonzero = [x for x in LPlist if x != 0]
        LP_threshold = np.percentile(LPlist_nonzero, LPtestPercentage)
        print(LP_threshold)
        L2testPercentage = j   # find the 10% tweets with largest projection onto topic space
        L2_threshold = np.percentile(L2list, L2testPercentage)
        print(L2_threshold)

        for t in range(0, Nt_test):
            if LPlist[t] < LP_threshold and L2list[t] > L2_threshold:

                # calculate the location of the tweet
                x = Spatial_test.iloc[t]["longitude"]
                y = Spatial_test.iloc[t]["latitude"]
                x_bin = math.floor((x - minlong) / hx)
                y_bin = math.floor((y - minlat) / hy)

                distribution = np.zeros((Nx, Ny))
                for T in range(0, NT):
                    distribution += (W_test_normal[t, T] * data[T]) # Calculate the distribution of the tweet

                if distribution.sum() > .1:
                    count += 1
                    print(t)
                    # find the bin with highest density function
                    indices = np.where(distribution == distribution.max())
                    x_bin_predict = indices[0][0]
                    y_bin_predict = indices[1][0]

                    # record the predictions and the tweet id in the variable "predict"
                   # predict[0].append(Spatial_test.iloc[t]["tweet_id"])
                   # predict[1].append(x_bin_predict)
                   # predict[2].append(y_bin_predict)

                    #print(np.absolute(x_bin - x_bin_predict))
                    #print(np.absolute(y_bin - y_bin_predict))

                    # calculate the distance from the correct bin
                    diffX = np.absolute(x_bin - x_bin_predict) * 180
                    diffY = np.absolute(y_bin - y_bin_predict) * 180
                    diff = math.sqrt(diffX**2 + diffY**2)
                    error += diff

                    if diff < 2000:
                        twokm += 1
                        if diff < 1000:
                            onekm += 1
                    if diffX == 0 and diffY == 0:
                        exact += 1


        print('the count is')
        print(count)
        print('percentage of test tweets predicting')
        print(count/Nt_test)
        print('average error is')
        print(error/count)  # the average error
        #pickle.dump(predict, open("predict.p", "wb"))


        print("percentage for exact is:")
        print(exact/count)

        print("percentage for 1km is:")
        print(onekm/count)

        print("percentage for 2km is:")
        print(twokm/count)

        output.append(np.array([i, j, count/Nt_test, exact/count, onekm/count, twokm/count]))
        print(i)
        print(j)
        with open('output.txt', 'a') as f:
                f.write(str(np.array([i, j, count/Nt_test, exact/count, onekm/count, twokm/count])) + '\n')
