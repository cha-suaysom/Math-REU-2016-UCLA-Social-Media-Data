if __name__ == "__main__": # sort of like with MPI, we need this to do multiprocessing on windows
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

#####################LOCATION MATRIX ##############################

    rows=100 #rows in grid of city
    cols = 100 #cols in grid of city
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
    Spatial_full = pickle.load(open('pandas_data_barc.pkl','rb'))
    Spatial_full = Spatial_full[Spatial_full["gps_precision"] == 10.0]
    Spatial_full = Spatial_full.head(100000)
    #location trimming
    print('hi')
    lat_upper_bound = 41.390205 + 2
    lat_lower_bound = 41.390205 - 2
    long_upper_bound = 2.154007 + 0.5
    long_lower_bound = 2.154007 - 0.5
    Spatial_full = Spatial_full[Spatial_full["latitude"] < lat_upper_bound]
    Spatial_full = Spatial_full[Spatial_full["latitude"] > lat_lower_bound]
    Spatial_full = Spatial_full[(Spatial_full["longitude"] < long_upper_bound)]
    Spatial_full = Spatial_full[(Spatial_full["longitude"] > long_lower_bound)]
    raw_text = Spatial_full["text"]
    maxlat = Spatial_full["latitude"].max()
    minlat = Spatial_full["latitude"].min()
    maxlong = Spatial_full["longitude"].max()
    minlong = Spatial_full["longitude"].min()
    print("bye")
    #SAVING GRID COORDINATES
    Xlist = []
    Ylist = []
    for row in Spatial_full.itertuples():
        # print(len(row))
        y = ((float(row[7]) - minlat) / (maxlat - minlat)) * (rows - 1 * 10 ** (-12))
        y = math.floor(y)
        x = ((float(row[8]) - minlong) / (maxlong - minlong)) * (cols - 1 * 10 ** (-12))
        x = math.floor(x)
        Xlist.append(x)
        Ylist.append(y)
    Spatial_full["xgrid"]= Xlist
    Spatial_full["ygrid"]= Ylist
    #creating a sample
    Spatial_full= Spatial_full.sample(frac = 1.0) ###Shuffle
    size = len(Spatial_full.index)
    pickle.dump(Spatial_full, open('full_pandas_data_barc.pkl', 'wb'))  # saves the rest of tweets for testing
    print(size)
    fraction = 0.5
    Spatial = Spatial_full.head(math.floor(size*fraction))
    rest_of_tweets_pandas= Spatial_full.tail(1+ math.floor(size*(1-fraction)))
    print(len(rest_of_tweets_pandas.index))
    pickle.dump(rest_of_tweets_pandas,open('rest_of_tweets_pandas_data_barc.pkl', 'wb')) #saves the rest of tweets for testing
    #Spatial = Spatial.head(500)
    #print(Spatial)
    #L #location matrix
    coorlist = []
    for row in Spatial.itertuples():
        x = row[10]
        y = row[11]
        coorlist.append(y*cols + x)
    length = len(coorlist)
    L = sps.vstack(vector_list[coorlist[i]] for i in range(0,length)) ### loops through all the tweets and adds the rows, L is created once.
    print(L.shape)





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


    print(full_text_tf_idf.shape)

    text_tf_idf = full_text_tf_idf[:math.floor(size*fraction),:]
    rest_of_tweets = full_text_tf_idf[-math.floor(1+size*fraction):,:]
    print(rest_of_tweets.shape)
    print(text_tf_idf.shape, rest_of_tweets.shape)

    pickle.dump(rest_of_tweets, open('rest_of_tweets_TFIDF_barc.pkl', 'wb'))#saves the rest of tweets for testing


############ CONCATENATING LOCATION AND TFIDF MATRICES ##############
    location_norm = sps.linalg.norm(L, 'fro')
    text_norm = sps.linalg.norm(text_tf_idf, 'fro')
    print(location_norm, text_norm, location_norm/text_norm)

    alpha = 0.1*(text_norm/location_norm) # Weight of location matrix, normalized so that text and location parts have the same frobinous norm
    L = alpha*L

    NMFLOC = sps.hstack((text_tf_idf, L))
    NMFLOC = NMFLOC.tocsr()
    print(NMFLOC.shape)

# ######### Exporting to MATLAB ######################
#
#     sio.savemat('TFIDF_Location_barcSample05', {'TF_IDF': NMF})
#     X = tf_idf.get_feature_names()
#     sio.savemat('voc_Location_barcSample05', {'TF_IDF_feature_names_barc10sample05' : X})


######## PYTHON NMF #############
    topic_model = NMF(n_components=100, verbose=1, tol=0.001)  # Sure lets compress to 100 topics why not...

    text_topic_model_W = topic_model.fit_transform(NMFLOC) # NMF's .transform() returns W by
    # default, but we can get H as follows:
    text_topic_model_H = topic_model.components_
    print("Topic Model Components:")
    print(text_topic_model_W[0]) # topic memberships of tweet 0
    print(len(text_topic_model_H[0]))
    print(text_topic_model_H[0]) # this is relative word frequencies within topic 0.
    # Maybe. We might need to to transpose this...

    text_topic_model_WH = (text_topic_model_W,text_topic_model_H)
    Topics = text_topic_model_W.argmax(axis=1)
    Spatial["topics"] = Topics.tolist()
    pickle.dump(Spatial, open('Location_pandas_data_barc.pkl', 'wb'))

    pickle.dump(tf_idf.get_feature_names(), open('TF_IDF_feature_names.pkl', 'wb'))
    pickle.dump(text_topic_model_WH, open('location_NMF_100_topics_barc_WH.pkl','wb'), protocol=4) # Save it to
    #pickle.dump(topic_model, open('NMF_vanc.pkl','wb'), protocol=4)
    # disk so we don't have to keep recalculating it later

