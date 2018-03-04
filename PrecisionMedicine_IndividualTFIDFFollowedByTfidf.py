#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 21:46:27 2017

@author: suresh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 22:29:29 2017

@author: suresh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 12:37:46 2017

@author: suresh
"""

from sklearn import *
import pandas as pd
import numpy as np
#import gensim
#from gensim import corpora
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import string
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import xgboost as xgb
import re
train = pd.read_csv('data/training_variants')
test = pd.read_csv('data/test_variants')
trainx = pd.read_csv('data/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('data/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
doc2VecDims=200
nGrams = 1
nTopWords=20000
test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values
def preprocess(text):
    text = text.lower()
    text.translate(str.maketrans('','',string.punctuation))
    return text
train['Text'] = train['Text'].map(preprocess)
test['Text'] = test['Text'].map(preprocess)
tfidfMat_dict = {}
tfidfVectorizer_dict = {}
for i in np.sort(np.unique(y)):
    print ("Fitting class {}".format(i))
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,nGrams), stop_words = 'english')
    tfidfVectorizer_dict[str(i)] = tf
    tfidfMat_dict[str(i)] =  tf.fit_transform(train.loc[lambda x : x.Class==i, "Text"])

def getTopFeatures(vectorizer, topN=100):
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names()
    top_features = [features[i] for i in indices[:topN]]
    return top_features

topWords = []
for i in np.sort(np.unique(y)):
    topWords.extend(getTopFeatures(tfidfVectorizer_dict[str(i)],nTopWords))
topWords = set(topWords)

def keepTopWords(text,*topWords):
    resultwords  = [word for word in text.split() if word in topWords]
    return ' '.join(resultwords)
def buildNonTextFeats(variations):
    temp = variations.copy()
    print('Encoding...')
    temp['Gene_Share'] = temp.apply(lambda r: sum([1 for w in r['Gene'].lower().split(' ') if w in r['Text'].split(' ')]), axis=1)
    temp['Variation_Share'] = temp.apply(lambda r: sum([1 for w in r['Variation'].lower().split(' ') if w in r['Text'].split(' ')]), axis=1)
    temp['Variation_First'] = temp.apply(lambda r: r['Variation'].lower()[0], axis=1)
    temp['Variation_Last'] = temp.apply(lambda r : r['Variation'].lower()[len(r['Variation'].lower())-1],axis=1)
    gen_var_lst = sorted(list(temp.Gene.unique()) + list(temp.Variation.unique()))
    gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
#    for i in range(np.shape(temp_lsa)[1]):
#        tempc.append('lsa'+str(i+1))
#    temp = pd.concat([temp, pd.DataFrame(temp_lsa, index=temp.index)], axis=1)
    for c in temp.columns:
        if temp[c].dtype == 'object':
            if c in ['Gene','Variation']:
                lbl = preprocessing.LabelEncoder()
                temp[c+'_lbl_enc'] = lbl.fit_transform(temp[c].values)  
                temp[c+'_len'] = temp[c].map(lambda x: len(str(x)))
                temp[c+'_words'] = temp[c].map(lambda x: len(str(x).split(' ')))
            elif c != 'Text':
                lbl = preprocessing.LabelEncoder()
                temp[c] = lbl.fit_transform(temp[c].values)
            if c=='Text': 
                temp[c+'_len'] = temp[c].map(lambda x: len(str(x)))
                temp[c+'_words'] = temp[c].map(lambda x: len(str(x).split(' '))) 
    var_loc = temp.apply(lambda r : re.findall('\d+',r['Variation']),axis=1)
    temp['Variation_Loc'] = [int(x[0]) if len(x)>0 else -999 for x in var_loc ]
    tempc = list(temp.columns)
    return temp, tempc
train = train.drop(['Class'], axis=1)
df_all = pd.concat((train, test), axis=0, ignore_index=True)
nonTextDF,colNames = buildNonTextFeats(df_all)
trainWithFeatures = nonTextDF.iloc[:len(train)]
testWithFeatures = nonTextDF.iloc[len(train):]
allText = trainWithFeatures['Text'].append(testWithFeatures['Text'],ignore_index=True)
# filter text
# fit tfidf and doc2vec on filtered text
def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

truncText= allText.apply(keepTopWords,args=(topWords))
sentences = constructLabeledSentences(allText)
doc2VecModel = Doc2Vec(min_count=1, window=2, size=doc2VecDims, sample=1e-4, negative=5, workers=8,iter=100,seed=1)

doc2VecModel.build_vocab(sentences)

doc2VecModel.train(sentences, total_examples=doc2VecModel.corpus_count, epochs=doc2VecModel.iter)

doc2VecModel.save('./docEmbeddings_nDim'+doc2VecDims+"_nGram"+nGrams+'_nTop'+nTopWords+'.d2v')

##allText['Text'] = allText['Text'].map(keepTopWords)
tfAll = TfidfVectorizer(analyzer='word', ngram_range=(1,1), stop_words = 'english',sublinear_tf=True)
tfidf_matrixAll =  tfAll.fit_transform(allText).todense()
train_arrays = np.zeros((train.shape[0], tfidf_matrixAll.shape[1]))
tfidfVecColNames =['DOC_'+(str(i)) for i in range(tfidf_matrixAll.shape[1])]
train_labels = np.zeros(train.shape[0])
for i in range(train.shape[0]):
    train_arrays[i] = tfidf_matrixAll[i]
    train_labels[i] = y[i]
#trainWithFeatures = trainWithFeatures.concat(pd.DataFrame(train_arrays),axis=1,ignore_index=True)
colsForTraining = [col for col in trainWithFeatures.columns if col not in ['ID','Text','Gene','Variation']]

trainWithFeaturesForTraining = trainWithFeatures[colsForTraining]
tfidfDFTrain = pd.DataFrame(train_arrays)
tfidfDFTrain.columns = tfidfVecColNames
trainWithFeaturesForTraining = pd.concat([trainWithFeaturesForTraining,tfidfDFTrain],axis=1,ignore_index=True)
trainWithFeaturesForTraining.columns = colsForTraining+tfidfVecColNames
test_arrays = np.zeros((test.shape[0], doc2VecDims))
for i in range(train.shape[0],allText.shape[0]):
    test_arrays[i-train.shape[0]] = tfidf_matrixAll[i]
tfidfDFTest = pd.DataFrame(test_arrays)
tfidfDFTest.columns = tfidfVecColNames
testWithFeaturesForPrediction = testWithFeatures[colsForTraining]
#concat was returning nan rows, hence the following dance
testWithFeaturesForPrediction.reset_index(drop=True,inplace=True)
tfidfDFTest.reset_index(drop=True,inplace=True)
testWithFeaturesForPrediction = pd.concat([testWithFeaturesForPrediction,tfidfDFTest],axis=1,ignore_index=True)
testWithFeaturesForPrediction.columns = colsForTraining+tfidfVecColNames
#train
denom = 0
fold = 10 #Change to 5, 1 for Kaggle Limits
y = y-1
for i in range(fold):
    params = {
        'eta': 0.03333,
        'max_depth': 4,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(trainWithFeaturesForTraining, y, test_size=0.18, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print(score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(testWithFeaturesForPrediction), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(testWithFeaturesForPrediction), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = test.ID
    submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)
preds /= denom
submission_xgb = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission_xgb['ID'] = test["ID"]
submission_xgb.to_csv('submission_individual_tfidf_overall_tfidf_'+"_"+str(nGrams)+"_"+str(nTopWords)+".csv", index=False)

