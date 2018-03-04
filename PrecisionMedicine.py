#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 12:37:46 2017

@author: suresh
"""

from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import pickle
#import gensim
#from gensim import corpora
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import string
train = pd.read_csv('data/training_variants')
test = pd.read_csv('data/test_variants')
trainx = pd.read_csv('data/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('data/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values
#stoplist = set('for a of the and to in'.split())
def getWordFrequencies(documents):
    f = open('stopwords.txt', 'r')
    stoplist = f.readlines()
    stoplist = set(' '.join(stoplist).split("\n "))
    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    #texts = [[token for token in text if frequency[token] > 1] for text in texts]
    #sortedDict = sorted(frequency.items(), key=lambda v : v[1],reverse=True)
    return frequency
def getWordFrequencies_v2(documents,topN=3000):
    f = open('stopwords.txt', 'r')
    stoplist = f.readlines()
    stoplist = set(' '.join(stoplist).split("\n "))
    wordList =list()
    for doc in documents:
        wordList+=[word for word in doc.lower().split() if word not in stoplist]
    wordfreq = [wordList.count(p) for p in wordList]
    return dict(zip(wordList,wordfreq))
#def tokenizer_stem_nostop(text):
#    porter = PorterStemmer()
#    return [porter.stem(w) for w in re.split('\s+', text.strip()) \
#            if w not in stop and re.match('[a-zA-Z]+', w)]
def preprocess(text):
    text = text.lower()
    text.translate(str.maketrans('','',string.punctuation))
    return text
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words = 'english',preprocessor=preprocess)
tfidf_matrix =  tf.fit_transform(train["Text"])
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)
def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf',probability=True)
    svm.fit(X, y)
    return svm
# Create and train the Support Vector Machine
svm = train_svm(X_train, y_train)

# Make an array of predictions on the test set
pred = svm.predict(X_test)

# Output the hit-rate and the confusion matrix for each model
print(svm.score(X_test, y_test))
print(confusion_matrix(pred, y_test))
test_tfidfd_matrix = tf.transform(test["Text"])
submission = svm.predict(test_tfidfd_matrix)
classPrediction = np.zeros((submission.size,submission.max()+1))
classPrediction[np.arange(submission.size), submission] = 1
subClass = pd.DataFrame(classPrediction)
del subClass[0]
subClass.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']
subClass.insert( 0,"ID",test["ID"]) 
subClass.to_csv("submission_rbf_recap_classPrediction.csv",index=False)
submission_prob = svm.predict_proba(test_tfidfd_matrix)
sub= pd.DataFrame(submission_prob)
sub.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']
sub.insert( 0,"ID",test["ID"]) 
sub.to_csv("submission_sublinear_tf.csv",index=False)
pickle.dump(tfidf_matrix,open("TrainTFIDFMatrix.pickle","wb"))
pickle.dump(test_tfidfd_matrix,open("TrainTFIDFMatrix.pickle","wb"))
svmClassifier = svm
pickle.dump(svm,open("TFIDF-SVM.pickle","wb"))
