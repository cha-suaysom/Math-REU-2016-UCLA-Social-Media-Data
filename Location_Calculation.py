import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
def LocationCalculation(name = 'barc', alpha = 0.5, number_of_topics = 100, training_fraction = 0.5):
    alpha = round(alpha, 1)
    training_fraction = round(training_fraction, 1)
    Spatial =  pickle.load(open(
        'Location_pandas_data_' + name + '_' + str(training_fraction) + 'training' + '_alpha' + str(alpha) + '.pkl',
        'rb'))
    NT = number_of_topics
    MSD_List = []#stores the Mean Square Distance between all the tweets in  agiven topic
    Topics_Size = []
    #Spatial = Spatial[Spatial["gps_precision"] == 10.0]#taking only location accurete tweets
    for T in range(0,NT):#Mean Square Distance Calculation: we want to find the sum of the square of the
        x = Spatial[Spatial["topics"] == T]# euclidean pairwise distances between all the tweets in each topic divided by the size of the topic (K)
        a = x["latitude"]
        b = x["longitude"]
        K =len(a)
        Topics_Size.append(K)
        X = np.array(a)#we calculate x axis pairwise square distances
        Y = np.array(b)#and then seperatley y axis distances
        MSD = (2/((K+1)**2))*((K*np.dot(X.T, X) - (X.sum())**2)+(K*np.dot(Y.T, Y) - (Y.sum())**2))#I am almost 100% this is correct but tell me if you guys see anything alarming
        MSD_List.append(MSD)#in the above, we used K+1 instead of K (Where K is the size of the topic) in the denomenator to make  sure we ar enot dividing by 0 for the empty topics
    df = pd.DataFrame(columns = ( "Length", "MSD"))#save the topics size and MSD into a data frame
    df["MSD"] = MSD_List
    df["Length"] = Topics_Size

    #next we want to create a "density function" of the tweets for each topic over the city of barcelona in order to calculate other location statistics
    #to do this we partition our location grid of [0,1]x[0,1] into LxL squares (here L = 100) we create matrices for each of the 500 topics to store the
    #the number of tweets in each of the squares, we later divide the matrix by the total number of tweets in the topic to get a probibistic matrix of
    # where are the tweets distrbuted for each topic
    if name  == "barc":
        rows = 100
        cols = 100
    elif name == "vanc":
        rows = 100
        cols = 180
    ArrayList = []
    for T in range(0,NT):#F density calculation
        A = np.zeros((cols,rows))
        G = Spatial[Spatial["topics"] == T]
        N = len(G.index)
        for row in G.itertuples():
            x = row[10]
            y= row[11]
            A[x,y] = A[x,y]+ 1
        ArrayList.append(A/N)
    TopicPeak = []
    for T in range(0,NT):
        B = ArrayList[T]
        C = B[:,:]
        max = 0
        xpeak = 0
        ypeak = 0
        for i in range(0,cols):
            for j in range(0,rows):
                if C[i][j] > max:
                    max = C[i,j]
                    xpeak = i
                    ypeak = j
        TopicPeak.append([xpeak,ypeak])
    df["peak"]= TopicPeak
    print(TopicPeak)

    #once we created the denisty function array we calculate the information theoretic entropy associated with the probablity distribution
    #as well as a fractional L^0.5 norm, normalized by the usual L^1 norm of the distribution
    MetricList= []
    EntropyList = []
    for X in ArrayList:
        L_P = ((np.sqrt(X)).sum()*(1/(rows*cols)))**2
        L_1 = (X.sum()+1*10**(-12))*(1/(rows*cols))
        Final_L = L_P/L_1# L^0.5 norm divided by L^1 norm
        MetricList.append(Final_L)
        Log = np.log(X)#Log(P) matrix
        Log = np.nan_to_num(Log)# removes enteries where P was 0 hence Log(P)  was undefined
        EntropyMatrix = np.multiply(X,Log)
        Ent = -EntropyMatrix.sum()#sum of P*log(P) where P is the probablity that for a certain topic the tweets are in the specific square
        EntropyList.append(Ent)
    df["L0.5"]= MetricList
    df["Entropy"] = EntropyList
    # df = df.sort_values(by = "MSD")
    # df = df[df["Length"]>0]#removes empty topics
    print(df.head(200))
    print("MSD Mean: " , df["MSD"].mean(),"L one half: ", df["L0.5"].mean())
    print("MSD Median: " , df["MSD"].median(),"L one half: ", df["L0.5"].median())
    #plt.plot(df["MSD"], df["L0.5"])
    # plt.scatter((df["L0.5"]), df["MSD"])
    # plt.xlabel("LP")
    # plt.ylabel("MSD")
    # plt.title("NMF500 Topics LP vs MSD values")
    # plt.show()
    pickle.dump(df, open('topic_stats_pandas_' + name + '_' + str(training_fraction) + 'training' + '_alpha' + str(alpha) + '.pkl',
        'wb'))
#LocationCalculation()
