#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 19:59:46 2017

@author: suresh
"""
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy
import pandas as pd

# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression
import str,string
train = pd.read_csv('data/training_variants')
test = pd.read_csv('data/test_variants')
trainx = pd.read_csv('data/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('data/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values

def cleanup(text):
    text = text.lower()
    text= text.translate(str.maketrans("","", string.punctuation))
    return text
train['Text'] = train['Text'].apply(cleanup)
test['Text'] = test['Text'].apply(cleanup)

