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

for name in raw_data['user']:
    tagDict[name] =[name] #Correct this later

i = 0
for index,row in raw_data.iterrows():
    # i = i+1
    # if (i >= 500000):
    #     break
    userName = row["user"]
    #print(userName)
    sentence = row["text"]
    #print(sentence)
    if (sentence.find('@') != -1):
        for word in sentence.split():
            if(word[0] == '@'):
                if (word[1:] in tagDict): #(word in tagDict):
                    #print(word)
                    tagDict[word[1:]].append(userName)
#Who got tagged the most


max_key = max(tagDict, key= lambda x: len(set(tagDict[x])))
print(max_key)
print(tagDict[max_key])

# wordList = '''awesome day of my life because i am great something some
# thing things unclear sun clear'''.split()
#
# wordOr = '|'.join(wordList)
#
# def splitHashTag(hashTag):
#   for wordSequence in re.findall('(?:' + wordOr + ')+', hashTag):
#     print ':', wordSequence
#     for word in re.findall(wordOr, wordSequence):
#       print word,
#     print
#
# for hashTag in '''awesome-dayofmylife iamgreat something
# somethingsunclear'''.split():
#   print '###', hashTag
#   splitHashTag(hashTag)
