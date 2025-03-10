# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:10:19 2016

@author: andy
"""
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF #, PCA, TruncatedSVD, LatentDirichletAllocation
import re  # regex
import scipy.io as sio
import scipy.sparse as sps
import numpy as np
import math
import pandas as pd
import sklearn.utils
import matplotlib.pyplot as plt
from sklearn import mixture
import os
Spatial = pickle.load(open('pandas_data_vanc.pkl','rb'))
Spatial = Spatial[Spatial["gps_precision"] == 10.0]   
 #################DEFINE CONSTANT###################################
#Barcelona
#rows = 100
#cols = 100
#LATITUDE_UPPER_BOUND = 41.390205 + 2
#LATITUDE_LOWER_BOUND = 41.390205 -2
#LONGITUDE_UPPER_BOUND= 2.154007 +0.5
#LONGITUDE_LOWER_BOUND = 2.154007 -0.5

#vancouver
rows = 100 #26KM
cols = 180 #49KM
LATITUDE_UPPER_BOUND = 49.2827 + 2
LATITUDE_LOWER_BOUND = 49.2827 - 2
LONGITUDE_UPPER_BOUND = -123.1207 + 2
LONGITUDE_LOWER_BOUND = -123.1207 - 2

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


#Define parameters here
n_trial = 1
perc_sample = 1
perc_train = 0.1
topic_reject_rank = 50

alp = 0.5
cis = [0.2,0.5]
LP_thresholds = np.linspace(100,10,10).astype(int).tolist()
#def start_predict(alp=0.5,ci=0.2,LP_threshold = 30, n_trial = 1):
    

for trial in np.arange(1,n_trial+1):
    current_directory = 'largeset/Vancouver/alpha' + str(alp).replace('.','') + '/trial' + str(trial) + '/' 
    #current_directory = 'largeset/Vancouver/trial' + str(trial).replace('.','') + '/alpha' + str(alp) + '/' 

    if not os.path.exists(current_directory):
        os.makedirs(current_directory)
        #os.makedirs(current_directory + 'gmms/')
    print("alpha = ", alp, "trial = ", trial)
    Sample= Spatial.sample(frac = perc_sample).copy()
    maxlat = Sample["latitude"].max()+10**(-12)
    minlat = Sample["latitude"].min()-10**(-12)
    maxlong = Sample["longitude"].max()+10**(-12)
    minlong = Sample["longitude"].min()-10**(-12)
    raw_text = Sample["text"]
    
    XGRID  = []
    YGRID = []
    for row in Sample.itertuples():
        y = ((float(row[7])-minlat)/(maxlat-minlat))*rows
        y = math.floor(y)
        x = ((float(row[8])-minlong)/(maxlong-minlong))*cols
        x = math.floor(x)
        XGRID.append(x)
        YGRID.append(y)
    Sample["xgrid"] = XGRID
    Sample["ygrid"] = YGRID
    
    fraction = perc_train
    length = len(Sample.index)
    Sample = Sample.sample(frac = 1.0)
    training_data = Sample.head(int(fraction*length))
    test_data = Sample.tail(length- int(fraction*(length)))
    
    #fraction = 1
    coorlist = []
    for row in training_data.itertuples():
        x = row[10] 
        y = row[11] 
        coorlist.append(y*cols + x)
    length = len(coorlist)
    L_full = sps.vstack(vector_list[coorlist[i]] for i in range(0,length)) ### loops through all the tweets and adds the rows, L is created once.
    L = sps.csr_matrix(L_full)
    pickle.dump(L_full, open(current_directory+'Location_matrix_full.pkl', 'wb'))

   

    #########TEXT MATRIX ##############
 
    # Make sure NaNs turn into strings
    # (We probably don't want this in the long run)
    raw_text_train = training_data["text"]
    raw_text_train = [str(x) for x in raw_text_train]
    #print("Number of Samples:", len(raw_text_train))
    print("Creating tfidf for training data")
    clean_text_train = [" ".join([   # joins a list of words back together with spaces in between them
                                re.sub(r'\W+', '', # force alphanumeric (after doing @ and # checks)
                                word.replace('"','').lower()) # force lower case, remove double quotes
                            for word in tweet.split() # go word by word and keep them if...
                                if len(word)>2 and # they are 3 characters or longer
                                not word.startswith('@') and # they don't start with @, #, or http
                                not word.startswith('#') and
                                not word.startswith('http')]
                            ).encode('ascii', errors='ignore') # force ascii encoding, ignore weird characters just in case
                        for tweet in raw_text_train]
    
    
    stop_words = [] # stop words file includes English, Spanish, and Catalan
    with open('stop_words.txt','r',encoding="utf-8") as f:
        stop_words = [word.replace("\n",'') for word in f.readlines()] # Have to remove \n's because I didn't copy the stop words cleanly
    
    #print("Stop word examples:", stop_words[:10])
    
    #print("\n----20 TWEETS----")
    # Lets make sure this looks right...
    #for tweet in clean_text[:20]: # First 20 tweets!
    #    print(tweet) # the b before these means they are ascii encoded
    #print("--------------")
    
    
    tf_idf_train = TfidfVectorizer(min_df=10,stop_words=stop_words, sublinear_tf= True)
    # min_df means ignore words that appear in less than that many tweets
    # we specify our stop words list here too
    
    full_text_tf_idf_train = tf_idf_train.fit_transform(clean_text_train) # like we talked about,
    # fit_transform is short hand for doing a .fit() then a .transform()
    # because 2 lines of code is already too much I guess...
    
    
    #print(full_text_tf_idf.shape)
    
    text_tf_idf_train = full_text_tf_idf_train
    #rest_of_tweets = full_text_tf_idf[int(fraction*length):,:]
    #print(rest_of_tweets.shape)
    #print(text_tf_idf.shape, rest_of_tweets.shape)
    
    #pickle.dump(rest_of_tweets, open(current_directory+'rest_of_tweets_TFIDF_barc.pkl', 'wb'))#saves the rest of tweets for testing
    
    
    ############ CONCATENATING LOCATION AND TFIDF MATRICES ##############
    location_norm = sps.linalg.norm(L, 'fro')
    text_norm = sps.linalg.norm(text_tf_idf_train, 'fro')
    #print(location_norm, text_norm, location_norm/text_norm)
    
    alpha = alp*(text_norm/location_norm) # Weight of location matrix, normalized so that text and location parts have the same frobinous norm
    L = alpha*L
    
    NMFLOC = sps.hstack((text_tf_idf_train, L))
    NMFLOC = NMFLOC.tocsr()
    #print(NMFLOC.shape)
    
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
    
    
    #%%
    ######## PYTHON NMF #############
    topic_model = NMF(n_components=100, verbose=1, tol=0.001)  # Sure lets compress to 100 topics why not...
    
    W = topic_model.fit_transform(NMFLOC) # NMF's .transform() returns W by
    # default, but we can get H as follows:
    HL = topic_model.components_
    #print("Topic Model Components:")
    #print(text_topic_model_W[0]) # topic memberships of tweet 0
    #print(len(text_topic_model_H[0]))
    #print(text_topic_model_H[0]) # this is relative word frequencies within topic 0.
    # Maybe. We might need to to transpose this...
    
    WHL = (W,HL)
    Topics = W.argmax(axis=1)
    training_data["topics"] = Topics
    Time = training_data["time"]
    training_data["date"] = pd.to_datetime(Time, infer_datetime_format= True)
    pickle.dump(training_data, open(current_directory+'training_data.pkl', 'wb'))
    
    pickle.dump(tf_idf_train.get_feature_names(), open(current_directory+'TF_IDF_feature_names.pkl', 'wb'))
    pickle.dump(WHL, open(current_directory+'location_NMF_100_topics_WHL.pkl','wb'), protocol=4) # Save it to
    #pickle.dump(topic_model, open('NMF_vanc.pkl','wb'), protocol=4)
    # disk so we don't have to keep recalculating it later
    
    
    
    #%%
    maxlat = training_data["latitude"].max()
    minlat = training_data["latitude"].min()
    maxlong = training_data["longitude"].max()
    minlong = training_data["longitude"].min()
    NT = 100 #number of topics
    MSD_List = []#stores the Mean Square Distance between all the tweets in  agiven topic
    Topics_Size = []
    #Spatial = Spatial[Spatial["gps_precision"] == 10.0]#taking only location accurete tweets
    for T in range(0,NT):#Mean Square Distance Calculation: we want to find the sum of the square of the
        x = training_data[training_data["topics"] == T]# euclidean pairwise distances between all the tweets in each topic divided by the size of the topic (K)
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
    msd_array = np.asarray(MSD_List)
    #%%
    pickle.dump(df, open(current_directory+'topic_stats_pandas.pkl', 'wb'))
    names = pickle.load(open(current_directory+'TF_IDF_feature_names.pkl','rb'))
    sorted_terms = []
    H = HL[:,0:HL.shape[1]-rows*cols]
    for x in H:
        sorted_term = []
        largest_10 = np.argsort(x)[-10:]
        for i in largest_10:
            sorted_term.append(names[i])
        sorted_terms.append(sorted_term)
    #sorted_terms = [list(names[np.argsort(x)[-10:].tolist()][::-1]) for x in H] # get the indices of the 10 highest values in each topic in H, then get the corresponding words for these values
 
    topicwords = ''
    for idx, s in enumerate(sorted_terms):
        #s = s.reverse()
        topicwords = topicwords+str(idx) + ' ' + str(s[::-1]) + '\n'
       # print(idx,s)
    text_file = open(current_directory + 'topicwords.txt', "w")
    text_file.write(topicwords)
    text_file.close()
   #%%
    #########TEXT MATRIX ##############
    print("Creating tfidf for test data")
    # Make sure NaNs turn into strings
    # (We probably don't want this in the long run)
    raw_text_test = test_data['text']
    raw_text_test = [str(x) for x in raw_text_test]
    #print("Number of Samples:", len(raw_text_test))
    
    clean_text_test = [" ".join([   # joins a list of words back together with spaces in between them
                                re.sub(r'\W+', '', # force alphanumeric (after doing @ and # checks)
                                word.replace('"','').lower()) # force lower case, remove double quotes
                            for word in tweet.split() # go word by word and keep them if...
                                if len(word)>2 and # they are 3 characters or longer
                                not word.startswith('@') and # they don't start with @, #, or http
                                not word.startswith('#') and
                                not word.startswith('http')]
                            ).encode('ascii', errors='ignore') # force ascii encoding, ignore weird characters just in case
                        for tweet in raw_text_test]
    
    stop_words = [] # stop words file includes English, Spanish, and Catalan
    with open('stop_words.txt','r',encoding="utf-8") as f:
        stop_words = [word.replace("\n",'') for word in f.readlines()] # Have to remove \n's because I didn't copy the stop words cleanly
    
   
    # Lets make sure this looks right...
    
    
    tf_idf_test = TfidfVectorizer(min_df=10,stop_words=stop_words, sublinear_tf= True,vocabulary = names)
   
    full_text_tf_idf_test = tf_idf_test.fit_transform(clean_text_test) # like we talked about,
   
    
    pickle.dump(full_text_tf_idf_test, open(current_directory+'testdata_TFIDF_barc.pkl', 'wb'))#saves the rest of tweets for testing
  
    full_text_tf_idf_test = sklearn.preprocessing.normalize(full_text_tf_idf_test)
    #ignore the location term
    normalized_H = sklearn.preprocessing.normalize(H,axis=1)
    
    
    topics = normalized_H*(full_text_tf_idf_test.T)
    pickle.dump(topics, open(current_directory+'test_topic_distribution.pkl', 'wb'))

    topic_distributions = topics.T
    Topic_list = (np.argmax(topics.T, axis = 1)).tolist()
    
    Time = test_data["time"]
    test_data["date"] = pd.to_datetime(Time, infer_datetime_format= True)
    test_data["topics"]= Topic_list
    pickle.dump(test_data, open(current_directory+'rest_of_tweets_pandas_data_barc.pkl','wb'))
#%%
    topic_reject = (MSD_List > np.percentile(MSD_List,topic_reject_rank))
    print("Learning GMMs")
    NT = 100#number of topics
    n_components_range = np.arange(5,40,5) # run through the number of components you want
    gmms =[]
    for T in range(0,NT):
        if topic_reject[T] == True:
            print("topic:" + str(T) + "rejected")
            gmms.append(mixture.GMM(0,'full'))
        else:
            print("topic:" + str(T))
            t1 = training_data[training_data["topics"] == T]
            a = t1["latitude"]
            b = t1["longitude"]
            K =len(a)
            lats = np.array(a)
            longs = np.array(b)
            locs = np.column_stack((longs*100,lats*100))
            #aic = [] # you can use either bic or aic. bic penalties n_components more than aic does
            lowest_bic = np.infty
            #bic_records = []
            for n_components in n_components_range:
                if (n_components < len(locs)):
                    #print("topic:" + str(T) + " components:" + str(n_components))
            
                    g = mixture.GMM(n_components, 'full')
                    g.fit(locs)
                    bic_now = g.bic(locs)
                    #bic_records.append(bic_now)
                    #aic.append(aic_now)
                    if bic_now<lowest_bic:
                        lowest_bic = bic_now
                        best_gmm = g
            gmms.append(best_gmm)
#==============================================================================
#                 plt.scatter(n_components_range,bic_records)
#                 plt.title('Topic:' + str(T) + ' emd_rank: ' +  str(np.where(emd_rank == T)[0][0]+1))
#                 plt.show()                
#==============================================================================
    for T in range(0,NT):
        if topic_reject[T] == True:
            print("topic:" + str(T) + "rejected")
            #pickle.dump(gmms[T], open(current_directory+'gmms/gmm_for_topic'+str(T)+'.pkl','wb'))
        else:
            print("topic:" + str(T))
            n = gmms[T].n_components
            t1 = training_data[training_data["topics"] == T]
            a = t1["latitude"]
            b = t1["longitude"]
            K =len(a)
            lats = np.array(a)
            longs = np.array(b)
            locs = np.column_stack((longs*100,lats*100))
            lowest_bic = gmms[T].bic(locs)
            best_gmm = gmms[T]
            #bic_records = []
            for n_components in np.arange(max(n-4,1),n+5):
                if (n_components < len(locs)):
                    #print("topic:" + str(T) + " components:" + str(n_components))
            
                    g = mixture.GMM(n_components, 'full', n_iter = 100)
                    g.fit(locs)
                    bic_now = g.bic(locs)
                    #bic_records.append(bic_now)
                    #aic.append(aic_now)
                    if bic_now<lowest_bic:
                        lowest_bic = bic_now
                        best_gmm = g
            gmms[T] = best_gmm
            #pickle.dump(best_gmm, open(current_directory+'gmms/gmm_for_topic'+str(T)+'.pkl','wb'))

#==============================================================================
#                 plt.scatter(np.arange(max(n-4,1),n+5),bic_records)
#                 plt.title('Topic:' + str(T) + ' emd_rank: ' +  str(np.where(emd_rank == T)[0][0]+1))
#                 plt.show()       
#==============================================================================
  
    for T in np.arange(0,100):
        if topic_reject[T] == False:
            if gmms[T].converged_ == False:
                print(T, "not converging, fixing")
                g = gmms[T]
                n_components = g.n_components
                g = mixture.GMM(n_components,'full', n_iter = 200)
                t1 = training_data[training_data["topics"] == T]
                a = t1["latitude"]
                b = t1["longitude"]
                K =len(a)
                lats = np.array(a)
                longs = np.array(b)
                locs = np.column_stack((longs*100,lats*100))
                g.fit(locs)
                gmms[T] = g
                #pickle.dump(g, open(current_directory+'gmms/gmm_for_topic_'+str(T)+'.pkl','wb'))
  #%%
    pickle.dump(gmms,open(current_directory + 'gmms.pkl','wb'))
    beta = 1/(msd_array.min()-msd_array.max())*math.log(0.1)
        #Generate the score

       
    score = np.exp((-1)*beta*msd_array)
    pickle.dump(score,open(current_directory + 'score_of_topics.pkl','wb'))
      
#==============================================================================
#             left_bounds = []
#             right_bounds = []
#             upper_bounds = []
#             lower_bounds = []
#              for t in np.arange(0,100):
#                  g = gmms[t]
#                  left_bounds.append(np.min(g.means_[:,0]))
#                  right_bounds.append(np.max(g.means_[:,0]))
#                  lower_bounds.append(np.min(g.means_[:,1]))
#                  upper_bounds.append(np.max(g.means_[:,1]))
#              left_bounds = np.asarray(left_bounds)                    
#              right_bounds = np.asarray(right_bounds)                    
#              lower_bounds = np.asarray(lower_bounds)                    
#              upper_bounds = np.asarray(upper_bounds)                    
# 
#==============================================================================
#%%    
    gmm_scores = []
    Lx = 180 #number of
    Ly = 100 #grids
    x = np.linspace(minlong*100,maxlong*100,Lx)
    y = np.linspace(minlat*100,maxlat*100,Ly)
    X,Y = np.meshgrid(x,y)
    XX = np.array([X.ravel(),Y.ravel()]).T
    for T in np.arange(0,100):
        if topic_reject[T] == False:
            P = np.exp(gmms[T].score(XX))
        else:
            P = np.zeros(Lx*Ly)
        gmm_scores.append(P)
    actual_locs=[]
    predicted_locs = []
    topics = []
    unif_Z = np.ones(Lx*Ly)
    unif_Z = unif_Z / (Lx*Ly)
    L_P_unif = ((np.sqrt(unif_Z)).sum())**2
    Frac_L = []
    #probs = []
    highest_ci = []
    for i in np.arange(0,len(test_data)):
        highest_ci.append(np.max(topic_distributions[i]))
    #testing_num = int(len(test_data))
    ci_thresholds = []
    for ci in cis:     
        if ci < 1:
            ci_threshold = ci
        else:
            ci_threshold = np.percentile(highest_ci,ci)
        ci_thresholds.append(ci_threshold)    
    ci_min = np.min(ci_thresholds)
    indicator_function_threshold = min(0.1,ci_min)-10**(-15)
    for i in np.arange(0,len(test_data)):
        if(i%50000 == 0):
            print("predicting",i)
        topic_distribution = topic_distributions[i]
        greatest_topic = np.argmax(topic_distribution)
        # Some of the tweet text consist characters that are not recognizable by ascii, and therefore its topic-distribution is (0,0,0...,0). We want to get rid of that
        if math.isnan(sum(topic_distribution)) == False and highest_ci[i]>=ci_min-10**(-15) and topic_reject[greatest_topic] == False: # You can change 0.0 to any number in [0,1)
            Z = np.zeros(Lx*Ly,)
            #norm_coeff = 0
            #prob = 0
#==============================================================================
#                     left = np.min(left_bounds[topic_distribution>0.1])
#                     right = np.max(right_bounds[topic_distribution>0.1])
#                     low = np.min(lower_bounds[topic_distribution>0.1])
#                     up = np.max(upper_bounds[topic_distribution>0.1])
#==============================================================================
            for t in np.arange(0,100):
                if topic_distribution[t]>=indicator_function_threshold and topic_reject[t] == False:
                    weight = score[t]*topic_distribution[t]
                    Z += weight*gmm_scores[t]
                    #norm_coeff += weight
                    #prob += weight*np.exp(gmms[t].score(actual))
#==============================================================================
#             Z = gmm_scores[greatest_topic]
#==============================================================================
            Z_reshaped = Z.reshape(X.shape)
            sumofZ = Z_reshaped.sum()
            if sumofZ == 0:
                result = 'rejected'
                Frac_L.append(L_P_unif)
                #probs.append('NA')
            else:
                result = XX[Z.argmax()]
                result = np.asarray([result[1]/100,result[0]/100])
                Z_reshaped = Z_reshaped/sumofZ
                L_P = ((np.sqrt(Z_reshaped)).sum())**2
                Frac_L.append(L_P)
                #prob = prob/norm_coeff
                #probs.append(prob)
            predicted_locs.append(result)
    
        else:
            result = 'rejected'
            predicted_locs.append(result)
            Frac_L.append(L_P_unif)
            #probs.append('NA')
        topics.append(greatest_topic)
        temp = test_data.iloc[i]
        actual = [temp['latitude'], temp['longitude']]
        actual = np.asarray(actual).reshape(1,-1)
        actual_locs.append(actual)
    predicted_results = pd.DataFrame(columns = ( "topic", "predicted_loc","actual_loc","Frac_L","highest_ci"))#,"probability"
#==============================================================================
#     for t in np.arange(0,len(test_data)):
#         temp = test_data.iloc[t]
#         actual = (temp['latitude'],temp['longitude'])
#         actual_locs.append(actual)
#==============================================================================
    predicted_results['topic'] = topics
    predicted_results['predicted_loc'] = predicted_locs
    predicted_results['actual_loc'] = actual_locs
    predicted_results['Frac_L'] = Frac_L
    predicted_results['highest_ci'] = highest_ci
    #predicted_results['probability'] = probs[0:testing_num]
    print("dumping results")
    pickle.dump(predicted_results,open(current_directory+ 'predict_results_msd.pkl','wb'), protocol = 4)
    print("Generating reports")
    rejected = predicted_results[predicted_results['predicted_loc']=='rejected']
    non_rejected = predicted_results[predicted_results['predicted_loc']!='rejected']   
    for k in np.arange(0,len(cis)):
        print(ci_thresholds[k])
        report = pd.DataFrame(columns = ("trainsize","testsize","ci_threshold","accepted_len","%accepted","LP_threshold%","good_len","%predict", "avgerr","mederr","200m","1km", "2km")) #"prob",
        
        accepted = non_rejected[non_rejected['highest_ci']>=ci_thresholds[k]]
        TRS = []
        TES = []
        
        CIT = []  
        AL = []
        ALP = []
        
        LPT = []
        GL = []
        GLP = []
        
        AE = []
        ME = []
        
        EXACT = []
        ONE = []
        TWO = []
        #PROB = []
        train_len = len(training_data)
        test_len = len(test_data)
                
        accepted_len = len(accepted)

        perc_accept = round(accepted_len/test_len*100,2)
        
        from geopy.distance import vincenty
        for LP_threshold in LP_thresholds:
            distances = []
            threshold_Frac_L = np.percentile(accepted['Frac_L'],LP_threshold)  
            for row in accepted.itertuples():
                if row[4] <= threshold_Frac_L: 
                    #print(t)        
                    actual = row[3]
                    predict = row[2]
                    distance = vincenty(actual,predict).km
                    distances.append(distance)
                else:
                    distances.append(9999)
            accepted['distance'] = distances.copy()
            good = accepted[accepted['distance'] != 9999]
            good_distances = good['distance'].tolist()
            good_len = len(good)
            perc_predict = round(good_len/test_len*100,2)
            avgerr = sum(good_distances)/good_len
            mederr = np.percentile(good_distances,50)
            
            count1 = 0
            count2 = 0
            count3 = 0
            for t in np.arange(0,len(good)):
                dist = good_distances[t]
                if dist < 0.2:
                    count1 += 1
                    count2 += 1
                    count3 += 1
                elif dist < 1: 
                    count2 += 1
                    count3 +=1
                elif dist < 2:
                    count3 +=1
            counts = [count1,count2, count3]
#==============================================================================
#             have_probs = non_rejected[non_rejected['probability'] != 'NA']
#             good_probs = have_probs[have_probs['Frac_L'] <= threshold_Frac_L]
#             list1 = good_probs['probability'].tolist()
#             list1 = [x for x in list1 if x!=0]
#             ln_probs = np.log(np.asarray(list1))
#             Np = len(list1)
#             llh = np.exp(np.sum(ln_probs)/Np)      
#==============================================================================
            
            TRS.append(train_len)
            TES.append(test_len)  
            
            CIT.append(round(ci_thresholds[k],2))
            AL.append(accepted_len)
            ALP.append(perc_accept)
            
            LPT.append(LP_threshold)         
            GL.append(good_len)
            GLP.append(perc_predict)
            
            AE.append(round(avgerr,3))
            ME.append(round(mederr,3))
            #PROB.append(round(llh,5))        
            percent1 = round(counts[0]/good_len*100,2)
            percent2 = round(counts[1]/good_len*100,2)
            percent3 = round(counts[2]/good_len*100,2)
            
            EXACT.append((counts[0],str(percent1)+"%"))
            ONE.append((counts[1],str(percent2)+"%"))
            TWO.append((counts[2],str(percent3)+"%"))         
            
        report['trainsize'] = TRS
        report['testsize'] = TES
            
        report['ci_threshold'] = CIT
        report['accepted_len'] = AL
        report['%accepted'] = ALP
            
        report['LP_threshold%'] = LPT
        report['good_len'] = GL
        report['%predict'] = GLP
        
        report['avgerr'] = AE
        report['mederr'] = ME
        #report['prob'] = PROB
        report['200m'] = EXACT
        report['1km'] = ONE
        report['2km'] = TWO
        
        extra_str = ""
       
        ci = cis[k]
        if ci<1:
            extra_str = "ci_" + str(ci).replace(".","") + "/"
        else:
            extra_str = "ci_" + str(ci).replace(".","") + "perc/"        
        if not os.path.exists(current_directory+extra_str):
            os.makedirs(current_directory+extra_str)
        
        
        pickle.dump(report,open(current_directory  + extra_str+ 'report.pkl','wb'))
        report.to_html(current_directory  + extra_str+ 'report.html' )

