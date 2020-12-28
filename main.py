import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def lower(data):
    lower = []
    for word in data:
        new = word.lower()
        lower.append(new)
    return lower

def punctuation(data):
    result = []
    table = str.maketrans('', '', string.punctuation)
    for word in data:
        new = word.translate(table)
        result.append(new)
    return result

def removeNumber(data):
    result = []
    for word in data:
        new = re.sub(r'\d+', '', word) 
        result.append(new)
    return result

def preprocessSearch(data):
    data = data.lower()
    table = str.maketrans('', '', string.punctuation)
    data = data.translate(table)
    data = re.sub(r'\d+', '', data)
    return data

def tf_idf(key, desc):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_weight = tfidf.fit_transform(desc)
    search = tfidf.transform([key])
    return search, tfidf_weight

def similarity(search, tfidf_weight):
    cosine_sim = cosine_similarity(search, tfidf_weight)
    sim = cosine_sim[0]
    most = []
    min = 5
    while min > 1:
        ind = np.argmax(sim)
        if(sim[ind] != 0):
            most.append(ind)
        sim[ind] = 0
        min -=1
    
    most = list(dict.fromkeys(most))
    return most
    

# Read Database
data = pd.read_csv('data.csv',encoding='cp1252')
data = data.dropna()
data['id'] = data.index
title = np.array(data['title'])
desc = np.array(data['description'])
code = np.array(data['code'])

# Preprocessing Data
lower = lower(desc)
punc = punctuation(lower)
num = removeNumber(punc)

# Ask for user input search
print("Search here:")
search = input()
search = preprocessSearch(search)

# Search in database for the most similar with the input
key, weight = tf_idf(search, num)
most = similarity(key,weight)

# Print search result and show the snippet code
x = code[most[0]]
print('\nResult for your search:')
for i in range(len(most)):
    print(i+1, title[most[i]])
print('\nWhich code do you want to see? (number)')
c = int(input())
print('\nJavascript snippet code:')
print(code[most[c-1]])


