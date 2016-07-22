import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF #, PCA, TruncatedSVD, LatentDirichletAllocation
import time
import re  # regex
import scipy.io as sio
import scipy.sparse as sps
import numpy as np
import math
import pandas as pd
import random
from copy import deepcopy
import sklearn.utils
from numpy import linalg
def LocationMatrix(name = 'barc', alpha = 0.5, number_of_topics = 100, training_fraction = 0.5):
    #################DEFINE CONSTANT###################################
    alpha = round(alpha, 1)
    training_fraction = round(training_fraction, 1)
    if(name == 'barc'):
        #Barcelona
        rows = 100
        cols = 100
        LATITUDE_UPPER_BOUND = 41.390205 + 2
        LATITUDE_LOWER_BOUND = 41.390205 -2
        LONGITUDE_UPPER_BOUND= 2.154007 +0.5
        LONGITUDE_LOWER_BOUND = 2.154007 -0.5
    if (name == 'vanc'):
        #vancouver
        rows = 100 #26KM
        cols = 180#49KM
        LATITUDE_UPPER_BOUND = 49.2827 + 2
        LATITUDE_LOWER_BOUND = 49.2827 - 2
        LONGITUDE_UPPER_BOUND = -123.1207 + 2
        LONGITUDE_LOWER_BOUND = -123.1207 - 2


    #####################LOCATION MATRIX ##############################
    def GenerateGrid(rows, cols, neighbor_weight):
        neighbor_weight = 0.5 ###first we create all of the location vectors
        nw = neighbor_weight
        vector_list = []
        for i in range(0,rows):
            for j in range(0, cols):
                A = np.zeros((rows,cols))
                for x in range(-1,2):
                    for y in range(-1,2):
                        if i+x<rows and i+x >-1 and j+y<cols and j+y >-1:
                            A[i+x][j+y]= nw
                A[i][j]=1
                B = sps.csr_matrix(A.reshape((1,rows*cols)))
                #print(B)
                vector_list.append(B)

        return vector_list
    #print(vector_list[0])
    vector_list = GenerateGrid(rows, cols, 0.5)

    Spatial = pickle.load(open('pandas_data_'+str(name)+'.pkl','rb'))
    Spatial = Spatial[Spatial["gps_precision"] == 10.0]
    Spatial = Spatial[Spatial["latitude"] < LATITUDE_UPPER_BOUND]
    Spatial = Spatial[Spatial["latitude"] > LATITUDE_LOWER_BOUND]
    Spatial = Spatial[(Spatial["longitude"] < LONGITUDE_UPPER_BOUND)]
    Spatial = Spatial[(Spatial["longitude"] > LONGITUDE_LOWER_BOUND)]
    maxlat = Spatial["latitude"].max()+10**(-12)
    minlat = Spatial["latitude"].min()-10**(-12)
    maxlong = Spatial["longitude"].max()+10**(-12)
    minlong = Spatial["longitude"].min()-10**(-12)
    print(minlat, maxlat)
    print(minlong, maxlong)
    #Spatial = Spatial.sample(frac=0.003)  ###Sample
    raw_text = Spatial["text"]
    XGRID  = []
    YGRID = []
    for row in Spatial.itertuples():
        y = ((float(row[7])-minlat)/(maxlat-minlat))*rows
        y = math.floor(y)
        x = ((float(row[8])-minlong)/(maxlong-minlong))*cols
        x = math.floor(x)
        XGRID.append(x)
        YGRID.append(y)
    Spatial["xgrid"] = XGRID
    Spatial["ygrid"] = YGRID

    fraction = training_fraction
    length = len(Spatial.index)
    Spatial = Spatial.sample(frac = 1.0)
    Spatial_sample = Spatial.head(int(fraction*length))
    rest_of_tweets_pandas = Spatial.tail(length- int(fraction*(length)))
    pickle.dump(rest_of_tweets_pandas, open('rest_of_tweets_pandas_data_'+name+'_'+str(training_fraction)+'training'+'_alpha'+str(alpha)+ '.pkl', 'wb'))  # saves the rest of tweets for testing
    raw_text = Spatial["text"]
    coorlist = []
    for row in Spatial.itertuples():
        x = row[10]
        y = row[11]
        coorlist.append(y*cols + x)
    length = len(coorlist)
    L_full = sps.vstack(vector_list[coorlist[i]] for i in range(0,length)) ### loops through all the tweets and adds the rows, L is created once.
    print(L_full.shape)
    #pickle.dump(L_full, open('Location_matrix_full_vanc.pkl', 'wb'))
    L = L_full[:int(fraction*length),:]
    # L_rand = L_full[:,:]
    # L_rand = sklearn.utils.shuffle(L_rand)
    # print(L_rand.shape)
    # L_rand = L_rand[(int((fraction*length))) :, :]
    #
    # print(L_rand.shape)
    # #print(L.shape)
    # L = sps.csr_matrix(L)
    # L_rand = sps.csr_matrix(L_rand)
    # #print(L_rand)
    # #print(L)
    # L_test = sps.vstack((L , L_rand))





#########TEXT MATRIX ##############





    # Make sure NaNs turn into strings
    # (We probably don't want this in the long run)
    raw_text = [str(x) for x in raw_text]
    print("Number of Samples:", len(raw_text))

    clean_text = [" ".join([   # joins a list of words back together with spaces in between them
                                re.sub(r'\W+', '', # force alphanumeric (after doing @ and # checks)
                                word.replace('"','').lower()) # force lower case, remove double quotes
                            for word in tweet.split() # go word by word and keep them if...
                                if len(word)>2 and # they are 3 characters or longer
                                not word.startswith('@') and # they don't start with @, #, or http
                                not word.startswith('#') and
                                not word.startswith('http')]
                            ).encode('ascii', errors='ignore') # force ascii encoding, ignore weird characters just in case
                        for tweet in raw_text]


    stop_words = [] # stop words file includes English, Spanish, and Catalan
    with open('stop_words.txt','r') as f:
        stop_words = [word.replace("\n",'') for word in f.readlines()] # Have to remove \n's because I didn't copy the stop words cleanly

    print("Stop word examples:", stop_words[:10])

    print("\n----20 TWEETS----")
    # Lets make sure this looks right...
    for tweet in clean_text[:20]: # First 20 tweets!
        print(tweet) # the b before these means they are ascii encoded
    print("--------------")


    tf_idf = TfidfVectorizer(min_df=10,stop_words=stop_words, sublinear_tf= True)
    # min_df means ignore words that appear in less than that many tweets
    # we specify our stop words list here too

    full_text_tf_idf = tf_idf.fit_transform(clean_text) # like we talked about,
    # fit_transform is short hand for doing a .fit() then a .transform()
    # because 2 lines of code is already too much I guess...


    #print(full_text_tf_idf.shape)

    text_tf_idf = full_text_tf_idf[:int(fraction*length),:]
    rest_of_tweets = full_text_tf_idf[int(fraction*length):,:]
    #print(rest_of_tweets.shape)
    #print(text_tf_idf.shape, rest_of_tweets.shape)

    pickle.dump(rest_of_tweets, open('rest_of_tweets_TFIDF_'+name+'_'+str(training_fraction)+'training'+'_alpha'+str(alpha)+ '.pkl', 'wb'))#saves the rest of tweets for testing


############ CONCATENATING LOCATION AND TFIDF MATRICES ##############
    location_norm = sps.linalg.norm(L, 'fro')
    text_norm = sps.linalg.norm(text_tf_idf, 'fro')
    print(location_norm, text_norm, location_norm/text_norm)

    adjustedAlpha = alpha*(text_norm/location_norm) # Weight of location matrix, normalized so that text and location parts have the same frobinous norm
    L = adjustedAlpha*L

    NMFLOC = sps.hstack((text_tf_idf, L))
    NMFLOC = NMFLOC.tocsr()
    print(NMFLOC.shape)

# ############## CONCATENATING LOCATION AND TFIDF MATRICES WITH TESTING ##################
#     location_norm = sps.linalg.norm(L_test, 'fro')
#     text_norm = sps.linalg.norm(full_text_tf_idf, 'fro')
#
#     alpha = 0.1 * (
#     text_norm / location_norm)  # Weight of location matrix, normalized so that text and location parts have the same frobinous norm
#     L = alpha * L
#     print(full_text_tf_idf.shape, L_test.shape)
#     NMFLOC = sps.hstack((full_text_tf_idf, L_test))
#     NMFLOC = NMFLOC.tocsr()
#     #print(NMFLOC.shape)



######## PYTHON NMF #############
    topic_model = NMF(n_components=number_of_topics, verbose=1, tol=0.001)  # Sure lets compress to 100 topics why not...

    text_topic_model_W = topic_model.fit_transform(NMFLOC) # NMF's .transform() returns W by
    # default, but we can get H as follows:
    text_topic_model_H = topic_model.components_
    print("Topic Model Components:")
    print(text_topic_model_W[0]) # topic memberships of tweet 0
    print(len(text_topic_model_H[0]))
    print(text_topic_model_H[0]) # this is relative word frequencies within topic 0.
    # Maybe. We might need to to transpose this...

    text_topic_model_WH = (text_topic_model_W,text_topic_model_H)


    pickle.dump(tf_idf.get_feature_names(), open('TF_IDF_feature_names_'+name+'_'+str(training_fraction)+'training'+'_alpha'+str(alpha)+ '.pkl', 'wb'))
    pickle.dump(text_topic_model_WH, open('location_NMF_'+str(number_of_topics)+'_topics_'+name+'_'+str(training_fraction)+'training'+'_alpha'+str(alpha)+ '.pkl','wb'), protocol=4) # Save it to
    #pickle.dump(topic_model, open('NMF_vanc.pkl','wb'), protocol=4)
    # disk so we don't have to keep recalculating it later

#Projection of tweets into the training topics

    Topics = text_topic_model_W.argmax(axis=1)


    pickle.dump(Spatial_sample, open('Location_pandas_data_' + name + '_' + str(training_fraction) + 'training' + '_alpha' + str(alpha) + '.pkl','wb'))
    Sps = pickle.load(open(
        'Location_pandas_data_' + name + '_' + str(training_fraction) + 'training' + '_alpha' + str(alpha) + '.pkl',
        'rb'))
    Sps["topics"] = Topics.tolist()
    pickle.dump(Sps, open(
        'Location_pandas_data_' + name + '_' + str(training_fraction) + 'training' + '_alpha' + str(alpha) + '.pkl',
        'wb'))

    H_text = text_topic_model_H[:, :-rows * cols]
    normalized_H = sklearn.preprocessing.normalize(H_text)
    print(np.linalg.norm((normalized_H[0:2, :]), 'fro'))
    print(normalized_H.shape, rest_of_tweets.shape)

    topics_guess = (normalized_H * (rest_of_tweets.T)).T  # estimates "W" assuming orthognality

    # calculates the topics using NLS problems
    # V_ = rest_of_tweets_TFIDF.T #V transpose
    # W_ = H_text.T #H transpsoe
    # H_ = topics_guess #W tranpose
    # (topics, somegrad, numberOfIterations) = _nls_subproblem(V_,W_,H_, 0.001,1000)
    # print("number of iteration for nls problem:",numberOfIterations)


    # essentially NLS problem, but under the NMF 'hood'
    NLS_solver = NMF(n_components=number_of_topics, init='custom', verbose=1.0, tol=0.005,
                      max_iter=1)  # Sure lets compress to 100 topics why not...
    NMF.nls_max_iter = 20
    # W_NMF = deepcopy(topics_guess)
    # topics = topic_model.fit_transform(rest_of_tweets_TFIDF, W =W_NMF.T, H = normalized_H)



    length = (topics_guess.shape)[0]
    #to reduce the strain on our machine we solve the NLS in blocks
    for i in range(0, 10):
        print(i)
        topics_guess_piece = topics_guess[(i * length) // 10:((i + 1) * length) // 10, :]
        rest_of_tweets_piece = rest_of_tweets[(i * length) // 10:((i + 1) * length) // 10, :]
        W_NMF = deepcopy(topics_guess_piece)
        topics_piece = NLS_solver.fit_transform(rest_of_tweets_piece, W=W_NMF, H=normalized_H)
        print(str(i) + ".5")
        if i == 0:
            topics = topics_piece
        else:
            topics = np.vstack((topics, topics_piece))
    print(topics.shape)
    pickle.dump(topics, open('test_topic_distribution_'+ name + '_' + str(training_fraction) + 'training' + '_alpha' + str(alpha) + '.pkl', 'wb'))
    distance = linalg.norm((topics - topics_guess), 'fro') / (linalg.norm(topics_guess, 'fro'))
    print(distance)
    topics = sklearn.preprocessing.normalize(topics)
    Topic_list = (np.argmax(topics, axis=1)).tolist()
    print(len(Topic_list))
    print(len(rest_of_tweets_pandas.index))

    pickle.dump(rest_of_tweets_pandas, open(
        'rest_of_tweets_pandas_data_' + name + '_' + str(training_fraction) + 'training' + '_alpha' + str(
            alpha) + '.pkl', 'wb'))
    rot = pickle.load(open(
        'rest_of_tweets_pandas_data_' + name + '_' + str(training_fraction) + 'training' + '_alpha' + str(
            alpha) + '.pkl', 'rb'))
    rot["topics"] = Topic_list
    pickle.dump(rot, open(
        'rest_of_tweets_pandas_data_' + name + '_' + str(training_fraction) + 'training' + '_alpha' + str(
            alpha) + '.pkl', 'wb'))
#LocationMatrix()
