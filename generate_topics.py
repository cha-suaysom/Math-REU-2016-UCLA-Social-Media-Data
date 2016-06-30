import pickle
import numpy as np

(W, H) = pickle.load(open('Location_NMF_100_topics_barc_WH.pkl','rb'))
names = np.array(pickle.load(open('TF_IDF_feature_names.pkl','rb')))

print(H.shape)
print(names[:500])
print(H[:20,:20])
H1 = H[:,:-10000]
sorted = [list(names[np.argsort(x)[-10:]][::-1]) for x in H1] # get the indices of the 10 highest values in each topic in H, then get the corresponding words for these values
for idx, s in enumerate(sorted):
    print(idx,s)


# Interesting topics!:
# ['follow', 'please', 'make', 'dream', 'babe', 'much', 'girl', 'back', 'world', 'amazing']
# ['buenas', 'noches', 'tardes', 'descanso', 'sueos', 'felices', 'besos', 'tods', 'buen', 'dulces'] - things said at night, like sweet dreams and such (sueos is dreams)
# ['love', 'much', 'amazing', 'really', 'everything', 'dont', 'guys', 'best', 'know', 'adore']
# ['lol', 'hahaha', 'dont', 'know', 'like', 'yeah', 'haha', 'omg', 'jugar', 'want']
# ['gente', 'buena', 'gusta', 'mucha', 'esa', 'encanta', 'mala', 'asco', 'suerte', 'quiere'] love and hate terms - asco is disgust, suerte is luck
# ['youre', 'idol', 'best', 'amazing', 'welcome', 'adore', 'music', 'plz', 'beginning', 'boyband']