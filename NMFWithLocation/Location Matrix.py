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
    import random
    from numpy import linalg as LA

#################DEFINE CONSTANT###################################
    rows = 100
    cols = 100
    maxlat = 41.390205 + 2
    minlat = 41.390205 -2
    maxlong= 2.154007 +0.5
    minlong = 2.154007 -0.5

#####################LOCATION MATRIX ##############################
    def GenerateGrid(rows, cols, neighbor_weight):
        #neighbor_weight = 0.5 ###first we create all of the location vectors
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
    Spatial = pickle.load(open('pandas_data_barc.pkl','rb'))
    Spatial= Spatial.sample(frac = 0.1) ###Sample traning data
    Spatial = Spatial[Spatial["gps_precision"] == 10.0]
    Spatial = Spatial[Spatial["latitude"] < maxlat]
    Spatial = Spatial[Spatial["latitude"] > minlat]
    Spatial = Spatial[(Spatial["longitude"] < maxlong)]
    Spatial = Spatial[(Spatial["longitude"] > minlong)]
    raw_text = Spatial["text"]

    #Return index of the training and those not in training data 
    def GenerateTrainingData(Spatial, fraction):     
        ##FIX THIS !!!

        #Spatial = Spatial.head(500)
        #print(Spatial)
        #L #location matrix
        coorlist = []
        for row in Spatial.itertuples():
            #print(len(row))
            y = ((float(row[7])-minlat)/(maxlat-minlat))*rows
            y = math.floor(y)
            x = ((float(row[8])-minlong)/(maxlong-minlong))*cols
            x = math.floor(x)
            coorlist.append(y*cols + x)

        length = len(coorlist)  
        #Generate random indices
        AllIndex = [i for i in list(range(0, length))]
        AllIndex = random.sample(AllIndex, length)
        knownIndex = AllIndex[:int(length*fraction)]
        unknownIndex = AllIndex[int(length*fraction):length]
 
        knownLocationMatrix = sps.vstack(vector_list[coorlist[i]] for i in list(range(0,length))) ### loops through all the tweets and adds the rows, L is created once.
  
        indexDict = {}
        for i in unknownIndex:
            indexDict[i] = 0
        for i in knownIndex:
            indexDict[i] = 1
        unknownVectorToStack = []
        for i in list(range(0,length)):
            if (indexDict[i] == 0):
                zeroMatrix = sps.csr_matrix(np.zeros(shape = (1, rows*cols)))
                unknownVectorToStack.append(zeroMatrix)
            else:
                unknownVectorToStack.append(vector_list[coorlist[i]])
        unknownLocationMatrix = sps.vstack(unknownVectorToStack)

        return (knownLocationMatrix, unknownLocationMatrix, knownIndex, unknownIndex)  

    knownLocationMatrix, unknownLocationMatrix, knownIndex, unknownIndex = GenerateTrainingData(Spatial, 0.9)
    print("KnownLocationMatrix")
    print(knownLocationMatrix)
    print("unknownLocationMatrix")
    print(unknownLocationMatrix)
    print("length knownindex")
    print(len(knownIndex))
    print("length unknownindex")
    print(len(unknownIndex))

    with open('KnownLocationMatrix.pkl', 'wb') as outfile:
        pickle.dump(knownLocationMatrix, outfile, protocol = 4)
    with open('unknownIndex.pkl', 'wb') as outfile:
        pickle.dump(unknownIndex, outfile, protocol = 4)
#     def oldAndNewLocation(KnownLocationNMFMatrix, W, H):
#         predictedLocationMatrix = W*H
#         #Obtain the row*col rightmost data to be LocationMatrix
#         knownLocationMatrix = sps.vstack(knownLocationMatrix[i] for i in unknownIndex)

#         #Obtain the row*col rightmost data to be predictedLocationMatrix
#         predictedLocationMatrix = sps.vstack(predictedLocationMatrix[i][(-row*col):] for i in unknownIndex)

#         return knownLocationMatrix, predictedLocationMatrix

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

    text_tf_idf = tf_idf.fit_transform(clean_text) # like we talked about,
    # fit_transform is short hand for doing a .fit() then a .transform()
    # because 2 lines of code is already too much I guess...

    print(text_tf_idf.shape)


############ CONCATENATING LOCATION AND TFIDF MATRICES ##############
    print("AlPHAAAAAAAAAAAAAAAAAAAAAAAAA")
    alpha = sps.linalg.norm(text_tf_idf,'fro')*1.0/sps.linalg.norm(unknownLocationMatrix,'fro') # Weight of location matrix
    unknownLocationMatrix = alpha*unknownLocationMatrix
    print(alpha)
    NMFLOC = sps.hstack((text_tf_idf, unknownLocationMatrix))
    NMFLOC = NMFLOC.tocsr()
    print(NMFLOC.shape)

# ######### Exporting to MATLAB ######################

    #sio.savemat('TFIDF_Location_barcSample05', {'TF_IDF': NMF})
    #X = tf_idf.get_feature_names()
    #sio.savemat('voc_Location_barcSample05', {'TF_IDF_feature_names_barc10sample05' : X})

    topic_model = NMF(n_components=100, verbose=1, tol=0.001)  # Sure lets compress to 100 topics why not...

    text_topic_model_W = topic_model.fit_transform(NMFLOC) # NMF's .transform() returns W by
    # # default, but we can get H as follows:
    text_topic_model_H = topic_model.components_
    print("Topic Model Components:")
    print(text_topic_model_W[0]) # topic memberships of tweet 0
    print(len(text_topic_model_H[0]))
    print(text_topic_model_H[0]) # this is relative word frequencies within topic 0.
    # # Maybe. We might need to to transpose this...

    text_topic_model_WH = (text_topic_model_W,text_topic_model_H)
    # Topics = text_topic_model_W.argmax(axis=1)
    # Spatial["topics"] = Topics
    # pickle.dump(Spatial, open('Location_pandas_data_vanc.pkl', 'wb'))

    pickle.dump(tf_idf.get_feature_names(), open('TF_IDF_feature_names_with_alpha.pkl', 'wb'))
    pickle.dump(text_topic_model_WH, open('location_NMF_100_topics_barc_WH_with_alpha.pkl','wb'), protocol=4) # Save it to
    # #pickle.dump(topic_model, open('NMF_vanc.pkl','wb'), protocol=4)
    # # disk so we don't have to keep recalculating it later