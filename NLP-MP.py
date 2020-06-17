# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:12:05 2020

@author: Sarvesh
"""

import os
import nltk
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


folder="C:/Users/Sarvesh/Desktop/SEM VIII/NLP/bbc/"
classes=['business','entertainment','politics','sport','tech']

print(os.listdir("path-to-folder/bbc/"))
content=[]
dataset_tar=[]
for idx,clas in enumerate(classes):
    path=folder+clas
    for filename in os.listdir(path):
        txt=open(path+'/'+filename,'r')
        for line in txt:
            if not line=='\n':
                ###regex preprocessing
                line=line.lower()
                line=re.sub("[^a-z]"," ",line)
                line=line.replace('bn','')
                line=re.sub("\s+"," ",line)
                line=re.sub("\s[a-z]\s"," ",line)
                content.append(line)
                dataset_tar.append(idx)
                

stop_words = stopwords.words('english')

dataset=[]
for idx,sent in enumerate(content):
    words=nltk.word_tokenize(sent)
    removed_sw = [x for x in words if x not in stop_words]
    string_formed=' '.join(removed_sw)
    dataset.append(string_formed)

# tf-idf
vectorizer = TfidfVectorizer(max_features=10000,max_df=0.05,min_df=10)

X = vectorizer.fit_transform(dataset)

lsa = TruncatedSVD(n_components=5,n_iter=5000)
lsa.fit(X)

terms = vectorizer.get_feature_names()

concept_words = {}

for i,comp in enumerate(lsa.components_):
    componentTerms=zip(terms,comp)
    sortedTerms=sorted(componentTerms,key=lambda x:x[1], reverse=True)
    sortedTerms=sortedTerms[:25]
    concept_words[i]=sortedTerms
    
###### Testing the model built

test_path = "path-to-folder/bbc_testset/"

test_content=[]
for filename in os.listdir(test_path):
    txt = open(test_path+filename,"r")
    curr_txt = txt.read()
    
    curr=curr_txt.lower()
    curr=re.sub("[^a-zA-Z]"," ",curr_txt)
    curr=curr.lower()
    curr=re.sub(r"\b[a-z]\b","",curr)
    curr=re.sub("\s+"," ",curr)
    test_content.append(curr)

test_dataset=[]
for sent in test_content:
    words=nltk.word_tokenize(sent)
    removed_sw = [x for x in words if x not in stop_words]
    string_formed=' '.join(removed_sw)
    test_dataset.append(string_formed)

### Actual testing now starts

for sent in test_dataset:
    concept_scores={}
    for i in range(5):
        concept_scores[i]=0
    
    words = nltk.word_tokenize(sent)
    for word in words:
        for k in concept_words.keys():
            for tup in concept_words[k]:
                if tup[0] == word:
                    concept_scores[k]+=tup[1]
                    break
    print(sent)
    print(concept_scores)
    #print(sorted(concept_scores.items(),key = lambda x:x[1], reverse=True))
    
import umap

X_topics = lsa.fit_transform(X)
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)

plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1], 
c = dataset_tar,
s = 10, # size
edgecolor='none'
)
plt.show()

