import pandas
import pickle

hashTagList = []
text = pickle.load(open('raw_text_data_vanc.pkl', 'rb'))
raw_data = pickle.load(open('pandas_data_vanc.pkl', 'rb'))
#
# times = 0
# for sentence in text:
#     times += 1
#     if (times > 1000):
#         break
#     if (sentence.find('#') != -1):
#         for word in sentence.split():
#             if (word[0] == '#'): #Change word to find
#                 print(word)
#                 hashTagList.append([word])

tagDict = {} #Key : users, Value : Users who tagged that user

#for name in raw_data['user']:
#    tagDict[name] =[name] #Correct this later

i = 0
for index,row in raw_data.iterrows():
    i = i+1
    if (i >= 1000):
        break
    userName = row["user"]
    print(userName)
    sentence = row["text"]
    print(sentence)
    if (sentence.find('@') != -1):
        for word in sentence.split():
            if(word[0] == '@'):
                if (True): #(word in tagDict):
                    print(word)
                    tagDict[word[1:]] = userName
print(tagDict)
