import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF  # , PCA, TruncatedSVD, LatentDirichletAllocation
import sklearn.preprocessing
import time
import pandas
import re  # regex
import scipy.io as sio
import scipy.sparse as sps
import numpy as np
import math
from scipy.stats.mstats import mode
import pandas as pd
from _nls_test import _nls_subproblem
from sklearn.decomposition import NMF
from scipy import linalg
from copy import deepcopy
def predictLocation(name = 'barc', alpha = 0.5, number_of_topics = 100, training_fraction = 0.5, threshold = 0.2):
    alpha = round(alpha,1)
    training_fraction = round(training_fraction, 1)
    threshold = round(threshold, 1)
    Topic_stats = pickle.load(open('topic_stats_pandas_' + name + '_' + str(training_fraction) + 'training' + '_alpha' + str(alpha) + '.pkl','rb'))
    rest_of_tweets_Data = pickle.load(open(
        'rest_of_tweets_pandas_data_' + name + '_' + str(training_fraction) + 'training' + '_alpha'+ str(
            alpha) + '.pkl', 'rb'))

    print("result For threshold: = " + str(threshold))
    topics  = pickle.load(open(
        'test_topic_distribution_' + name + '_' + str(training_fraction) + 'training' + '_alpha' + str(alpha) + '.pkl',
        'rb'))
    B = (np.max(topics, axis=1) >= threshold)
    halfkmlist = []
    onekmlist = []
    twokmlist = []
    exactlist = []
    lengthList = []
    selected_tweets_data = rest_of_tweets_Data[B]

    for T in range(0, number_of_topics):
        subset = selected_tweets_data[selected_tweets_data["topics"] == T]
        xgrid = subset["xgrid"].tolist()
        ygrid = subset["ygrid"].tolist()
        A = np.asarray((Topic_stats["peak"]))
        # print(mode(xgrid, axis = None))
        # print(mode(ygrid, axis = None))
        # print((A[T])[0], (A[T])[1])
        length = len(xgrid)
        lengthList.append(length)
        exact = 0
        halfkm = 0
        oneKm = 0
        twoKm = 0
        if(name == 'barc'):
            gridlength = 180
        elif(name == 'vanc'):
            gridlength = 260
        for i in range(0, length):
            Dis = math.sqrt((xgrid[i] - A[T][0]) ** 2 + (ygrid[i] - A[T][1]) ** 2) * gridlength
            if Dis == 0:
                exact += 1
            if Dis < 500:
                halfkm = halfkm + 1
            if Dis < 1000:
                oneKm = oneKm + 1
            if Dis < 2000:
                twoKm = twoKm + 1

        halfkmlist.append(halfkm)
        exactlist.append(exact)
        onekmlist.append(oneKm)
        twokmlist.append(twoKm)
    # print(len(Topic_stats.index))
    # print(len(onekmlist))
    Topic_stats["Exact"] = exactlist
    Topic_stats["500M"] = halfkmlist
    Topic_stats["1KM"] = onekmlist
    Topic_stats["2KM"] = twokmlist
    Topic_stats["TestingLength"] = lengthList
    numberOfLoops = 10
    C = np.array(range(numberOfLoops , 1, -1))
    C = number_of_topics * C/numberOfLoops
    prediction_stats =  pd.DataFrame(columns = ( "number of topics", "training fraction", "alpha", "threshold",
                                                 "exact", "500M", "1KM", "2KM" ))#save the topics size and MSD into a data fra
    numberOfLoops = numberOfLoops -1
    prediction_stats["number of topics"]  = [number_of_topics] * numberOfLoops
    prediction_stats["training fraction"] = [training_fraction] * numberOfLoops
    prediction_stats["alpha"] = [alpha] * numberOfLoops
    prediction_stats["threshold"] = [threshold] * numberOfLoops
    numberOfLoops = numberOfLoops + 1


    MSD_PctList = []
    exactAverageList = []
    halfkmAverageList = []
    onekmAverageList = []
    twokmAverageList = []
    PctPredict = []
    for numberOfTopics in C:
        sortedTopics = Topic_stats.sort_values(by="MSD").head(int(numberOfTopics))
        MSD_PctList.append(100*(numberOfTopics/number_of_topics))
        TotalTestingLength = np.sum(sortedTopics["TestingLength"].tolist())
        print("Percentage That we Predicted:")
        pp = 100.00 * TotalTestingLength / len(rest_of_tweets_Data)
        PctPredict.append(pp)
        print("Number of Topics by MSD")
        print(numberOfTopics)
        print("TotalLength")
        print(TotalTestingLength)
        print("Alltweet")
        print(len(rest_of_tweets_Data))
        exactAverage = np.sum(sortedTopics["Exact"].tolist()) * 100 / TotalTestingLength
        exactAverageList.append(exactAverage)
        print("ExactAverage")
        print(exactAverage)
        print("500M")
        halfkmAverage = np.sum(sortedTopics["500M"].tolist()) * 100 / TotalTestingLength
        halfkmAverageList.append(halfkmAverage)


        oneKmAverage = np.sum(sortedTopics["1KM"].tolist()) * 100 / TotalTestingLength
        print("1KMAverage")
        print(oneKmAverage)
        onekmAverageList.append(oneKmAverage)
        twoKmAverage = np.sum(sortedTopics["2KM"].tolist()) * 100 / TotalTestingLength
        print("2KMAverage")
        print(twoKmAverage)
        twokmAverageList.append(twoKmAverage)
        print("  ")
    print(len(prediction_stats.index), len(MSD_PctList))

    prediction_stats["MSD Pct"] = MSD_PctList
    prediction_stats["pct predicted"] = PctPredict
    prediction_stats["exact"] = exactAverageList
    prediction_stats["500M"] = halfkmAverageList
    prediction_stats["1KM"] = onekmAverageList
    prediction_stats["2KM"] = twokmAverageList

    return prediction_stats
#predictLocation()