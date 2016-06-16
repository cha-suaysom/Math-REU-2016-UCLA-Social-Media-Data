import pandas as pd
import pickle
import numpy as np
import  datetime as dt
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import date
(W, H) = pickle.load(open('NMF_500_topics_WH.pkl','rb'))
names = np.array(pickle.load(open('TF_IDF_feature_names.pkl','rb')))
print(W.shape)
print(H.shape)
Spatial = pickle.load(open('pandas_data.pkl','rb'))
Topics = W.argmax(axis=1)
Spatial["topics"] = Topics
Time = Spatial["time"]
Spatial["date"] = pd.to_datetime(Time, infer_datetime_format= True)
start = dt.date(2014,9,1)
end = dt.date(2014,11,30)
N =1.0 +(end-start).days
TemporalList= []
#Obtaining a temporal distribution function
for T in range(0,500):
    TopicDf= Spatial[Spatial["topics"]== T]
    datelist = TopicDf["date"].tolist()
    K= len(datelist)
    A= np.zeros(N)
    for i in range(0,K):
        t = (datelist[i].date()- start).days
        A[t] = A[t] + 1
    A= A/K
    TemporalList.append(A)
#Fractional L0.5 value and Entropy calculation
EntropyList = []
L_list = []
Prevelant_date = []
Variance_List = []
for T in range(0,500):
    # Entropy calculation
    X = TemporalList[T]
    Log = np.log(X)
    Log = np.nan_to_num(Log)
    Ent = np.multiply(Log, X)
    EntropyList.append((-Ent.sum()))
    #L0.5 calculation
    L_P = ((np.sqrt(X)).sum()*(1/N))**2
    L_1 = X.sum()/N
    L_list.append(((L_P)/(L_1 +1*10**(-12))))
    #Variance
    E = np.arange(N)
    square = np.multiply(E,E)
    var = np.dot(X,square.T)-(np.dot(X,E.T))**2
    var = var/((X.sum())**2)
    Variance_List.append(var)
    #prev. date
    change = int(np.argmax(X))
    popular= start + dt.timedelta(days = change)
    Prevelant_date.append(popular)

df= pd.DataFrame(columns= ("entropy", "L0.5"))
df["entropy"]= EntropyList
df["L0.5"] = L_list
df["var"]= Variance_List
df["prev. Date"]= Prevelant_date
df = df.sort_values(by = "var")
print(df)
plt.scatter(df["entropy"], df["var"])
plt.xlabel("entropy")
plt.ylabel("L0.5")
plt.title("NMF500 Topics LP vs MSD values")
plt.show()






