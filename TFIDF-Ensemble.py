#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:05:15 2017

@author: suresh
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn import *
from gensim.models import Doc2Vec
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Dropout,Input,merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1,l2,l1_l2
from keras.utils import np_utils
from keras.optimizers import SGD,Adam
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import backend as K 
import matplotlib.pyplot as plt
from Siamese_Loader import Siamese_Loader
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold;
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string
train_var = pd.read_csv('data/training_variants')
train_var1=pd.read_csv('data/test_variants')
train_text = pd.read_csv('data/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train_text1 = pd.read_csv('data/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train_var1_class = pd.read_csv('data/stage1_solution_filtered.csv')
train_var1_class_np = train_var1_class.values
train_var1_class_ids = np.argmax((train_var1_class_np[:,1:]),axis=1)+1
train_var1_with_class_ids = train_var1[train_var1['ID'].isin(train_var1_class['ID'])]
train_text1_with_class_ids = train_text1[train_text1['ID'].isin(train_var1_class['ID'])]
train_var1_with_class_ids['Class'] =train_var1_class_ids
train_var1_with_class_ids_from_stage1_test = train_var1_with_class_ids[~train_var1_with_class_ids['ID'].isin(train_var['ID'])]
train_text1_with_class_ids_from_stage1_test = train_text1_with_class_ids[~train_text1_with_class_ids['ID'].isin(train_var['ID'])]
train_var = pd.concat([train_var,train_var1_with_class_ids_from_stage1_test]) 
train_text = pd.concat([train_text,train_text1_with_class_ids_from_stage1_test])
#load test data
test_var = pd.read_csv('data/stage2_test_variants.csv')
test_text = pd.read_csv('data/stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
gene_class = pd.read_csv('data/Census_all.csv',sep=",")
gene_colnames = ['Gene Symbol','Hallmark','Molecular Genetics','Role in Cancer','Mutation Types']
gene_class_trim = gene_class[gene_colnames]
embeddingDim = 500
#train = pd.merge(train_var, train_text, how='left', on='ID').fillna('')
#y = train['Class'].values
##train = train.drop(['Class'], axis=1)
#
#test = pd.merge(test_var, test_text, how='left', on='ID').fillna('')
#pid = test['ID'].values
#train_var["HallMarkGene"] = train_var.apply(lambda r: 1 if gene_class_trim.loc[(gene_class_trim["Gene Symbol"]==r['Gene']).any() ]["Hallmark"]=="Yes" else 0, axis=1)
train_var_withFeatures = pd.merge(train_var,gene_class_trim, how="left",left_on="Gene",right_on="Gene Symbol")
def getGeneFeatures(df):
    df["Hallmark"] = np.where(df['Hallmark']=='Yes', 1, 0)
    df["Gene_Trans"] = np.where(df['Mutation Types'].str.contains('T'), 1, 0)
    df["Gene_Missense"] = np.where(df['Mutation Types'].str.contains('Mis'), 1, 0)
    df["Gene_Nonsense"] = np.where(df['Mutation Types'].str.contains('N'), 1, 0)
    df["Gene_Frameshift"] = np.where(df['Mutation Types'].str.contains('F'), 1, 0)
    df["Gene_Amplification"] = np.where(df['Mutation Types'].str.contains('A'), 1, 0)
    df["Gene_Deletion"] = np.where(df['Mutation Types'].str.contains('D'), 1, 0)
    df["Gene_Splicesite"] = np.where(df['Mutation Types'].str.contains('S'), 1, 0)
    df["Gene_Promoter"] = np.where(df['Mutation Types'].str.contains('Promoter'), 1, 0)
    df["Gene_Other"] = np.where(df['Mutation Types'].str.contains('O'), 1, 0)
    df["Oncogene"] = np.where(df['Role in Cancer'].str.contains('oncogene'), 1, 0)
    df["Fusion"] = np.where(df['Role in Cancer'].str.contains('fusion'), 1, 0)
    df["TSG"] = np.where(df['Role in Cancer'].str.contains('TSG'), 1, 0)
    df["Dominant"] = np.where(df['Molecular Genetics'].str.contains('Dom'), 1, 0)
    df["Recessive"] = np.where(df['Molecular Genetics'].str.contains('Rec'), 1, 0)
    df["XLinked"] = np.where(df['Molecular Genetics'].str.contains('X'), 1, 0)
    df['Variation_First'] = df.apply(lambda r: r['Variation'].lower()[0], axis=1)
    df['Variation_Last'] = df.apply(lambda r : r['Variation'].lower()[len(r['Variation'].lower())-1],axis=1)
    df['AAChange'] = df['Variation_First']+train_var_withFeatures['Variation_Last']
    return df
train_var_withFeatures = getGeneFeatures(train_var_withFeatures)
##prepare test data
test_var_withFeatures = pd.merge(test_var,gene_class_trim, how="left",left_on="Gene",right_on="Gene Symbol")
test_var_withFeatures = getGeneFeatures(test_var_withFeatures )
####
## amino acid and letter codes
# alaninie ala A
# aspargine asx B
# cysteine cys C
# aspartic acid asp D
# glutamic acid glu E
# phynylalanine phe F
# glycine gly G
# histidine his H
# isoleucine ile I
# leucine leu L
# lysine lys K
# methionine met M
# asparagine asn N
# proline pro P
# glutamine gln Q
# arginine arg R
# serine ser S
# theronine thr T
# valine val V
# tryptophan trp W
# tyrosine tyr Y
# glutamic acid glx Z
####
def isAliphatic (letter) :
    if (letter.lower() in ['a','i','l','m','v']):
        return 1
    return 0
def isAromatic (letter) :
    if (letter.lower() in ['f','w','y']):
        return 1
    return 0
def isNeutral(letter):
    if (letter.lower() in ['n','c','q','s','t']):
        return 1
    return 0
def isAcidic(letter):
    if (letter.lower() in ['d','e']):
        return 1
    return 0
def isBasic(letter):
    if (letter.lower() in ['r','h','k']):
        return 1
    return 0
def isUnique(letter):
    if (letter.lower() in ['g','p']):
        return 1
    return 0
def isCharged(letter):
    if (letter.lower() in ['r','k','d','e']):
        return 1
    return 0
def isPolar(letter):
    if (letter.lower() in ['q','n','h','s','t','y','c','w']):
        return 1
    return 0
def isHydrophobic(letter):
    if (letter.lower() in ['a','i','l','m','v','f','v','p','g']):
        return 1
    return 0

def getVariantFeatures(df):
    df["FirstAliphatic"] = df['Variation_First'].apply(isAliphatic)
    df["LastAliphatic"] = df['Variation_Last'].apply(isAliphatic)
    df["FirstAromatic"] = df['Variation_First'].apply(isAromatic)
    df["LastAromatic"] = df['Variation_Last'].apply(isAromatic)
    df["FirstNeutral"] = df['Variation_First'].apply(isNeutral)
    df["LastNeutral"] = df['Variation_Last'].apply(isNeutral)
    df["FirstAcidic"] = df['Variation_First'].apply(isAcidic)
    df["LastAcidic"] = df['Variation_Last'].apply(isAcidic)
    df["FirstBasic"] = df['Variation_First'].apply(isBasic)
    df["LastBasic"] = df['Variation_Last'].apply(isBasic)
    df["FirstUnique"] = df['Variation_First'].apply(isUnique)
    df["LastUnique"] = df['Variation_Last'].apply(isUnique)
    df["FirstCharged"] = df['Variation_First'].apply(isCharged)
    df["LastCharged"] = df['Variation_Last'].apply(isCharged)
    df["FirstPolar"] = df['Variation_First'].apply(isPolar)
    df["LastPolar"] = df['Variation_Last'].apply(isPolar)
    df["FirstHydrophobic"] = df['Variation_First'].apply(isHydrophobic)
    df["LastHydrophobic"] = df['Variation_Last'].apply(isHydrophobic)
    return df
train_var_withFeatures = getVariantFeatures(train_var_withFeatures)
test_var_withFeatures = getVariantFeatures(test_var_withFeatures)
engineeredFeatures = ['Gene_Trans','Gene_Missense', 'Gene_Nonsense', 'Gene_Frameshift','Gene_Amplification', 
                      'Gene_Deletion', 'Gene_Splicesite', 'Gene_Promoter', 'Gene_Other',
                      'Oncogene','Fusion','TSG','Dominant','Recessive','XLinked',
                      'FirstAliphatic','LastAliphatic','FirstAromatic','LastAromatic',
                      'FirstNeutral','LastNeutral','FirstAcidic','LastAcidic',
                      'FirstBasic','LastBasic','FirstUnique','LastUnique',
                      'FirstCharged','LastCharged','FirstPolar','LastPolar',
                      'FirstHydrophobic','LastHydrophobic']#+list(aachangepd.columns.values)
trainWithFeaturesForTraining = train_var_withFeatures[engineeredFeatures]
testWithFeaturesForTesting = test_var_withFeatures[engineeredFeatures]
# tf-idf features from The-Owl script on Kaggle
train = pd.merge(train_var, train_text, how='left', on='ID').fillna('')
y = train['Class'].values
train = train.drop(['Class'], axis=1)

test = pd.merge(test_var, test_text, how='left', on='ID').fillna('')
pid = test['ID'].values
df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)
for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 

train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]
del(df_all)
def tokenizer_stem_nostop(text):
    porter = PorterStemmer()
    return [porter.stem(w) for w in re.split('\s+', text.strip()) \
            if w not in stop and re.match('[a-zA-Z]+', w)]
def preprocess(text):
    text = text.lower()
    text.translate(str.maketrans('','',string.punctuation))
    return text
nltk.download('stopwords')
stop = stopwords.words('english')
#vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),stop_words='english', sublinear_tf=True, max_features=1000,
#                             preprocessor=preprocess, tokenizer=tokenizer_stem_nostop)
#tfidf_features = vectorizer.fit_transform(train_text['Text']).toarray()
class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)

print('Pipeline...')
fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char',max_features=50, ngram_range=(1, 2))), ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', max_features=50,ngram_range=(1, 2))), ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            #commented for Kaggle Limits
            ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(analyzer='word', ngram_range=(1, 2),stop_words='english', sublinear_tf=True, max_features=1000,
                             preprocessor=preprocess, tokenizer=tokenizer_stem_nostop)), ('tsvd3', decomposition.TruncatedSVD(n_components=500, n_iter=25, random_state=12))]))
        ])
    )])

train = fp.fit_transform(train); print(train.shape)
test = fp.transform(test); print(test.shape)
train_np = np.concatenate((train,trainWithFeaturesForTraining.values),axis=1)
test_np = np.concatenate((test,testWithFeaturesForTesting.values),axis=1)
#train_doc2Vec = np.zeros((train_var.shape[0],embeddingDim ))
##get doc2Vec features
#model = Doc2Vec.load('/media/suresh/520beb03-066a-471e-bdc1-e91782049d99/kaggle/PersonalizedMedicine/docEmbeddings_500.d2v')
#for i in range(train_var.shape[0]):
#    train_doc2Vec[i] = model.docvecs['Text_'+str(i)]
#test_doc2Vec = np.zeros((test_var.shape[0], embeddingDim))
#for i in range(train_var.shape[0],train_var.shape[0]+test_var.shape[0]):
#    test_doc2Vec[i-train_var.shape[0]] = model.docvecs['Text_'+str(i)]
#train_np = np.concatenate((train_np,train_doc2Vec),axis=1)
#test_np = np.concatenate((test_np,test_doc2Vec),axis=1)
#
def train_xgboost(trainWithFeaturesForTraining,y,fold,test):
    if np.min(y)!=0:
        y = y-1
    denom = 0
    fold = fold #Change to 5, 1 for Kaggle Limits
    #y = y-1
    for i in range(fold):
        params = {
            'eta': 0.01,
            'max_depth': 5,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': 9,
            'seed': i,
            'silent': True
        }
        x1, x2, y1, y2 = model_selection.train_test_split(trainWithFeaturesForTraining, y, test_size=0.18, random_state=i)
        watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
        predictions = model.predict(xgb.DMatrix(x2))
        predictions = np.argmax(predictions,axis=1)
        print("accurcy = {}".format(metrics.accuracy_score(y2, predictions)))
        score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
        print(score1)
        if denom != 0:
            pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
            preds += pred
        else:
            pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit+80)
            preds = pred.copy()
        denom += 1
    preds /= denom
    return preds
xgb_preds = train_xgboost(train_np,train_var['Class'].values,10,test_np)
submission = pd.DataFrame(xgb_preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('Predictions/stage2/submission_tfidf_gene_features.csv', index=False)
##ensembling
x1, x2, y1, y2 = model_selection.train_test_split(train_np, train_var['Class'].values, test_size=0.15, random_state=1)
ntrain = x1.shape[0]
ntest = test_np.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def predict_proba(self,x):
        return self.clf.predict_proba(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
        
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.01
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 7,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x1, y1-1, test_np) # Extra Trees
print("random forest classifier...")
rf_oof_train, rf_oof_test = get_oof(rf,x1, y1-1, test_np) # Random Forest
print("adaboost classifier...")
ada_oof_train, ada_oof_test = get_oof(ada, x1, y1-1, test_np) # AdaBoost 
print("gradient boost classifier...")
gb_oof_train, gb_oof_test = get_oof(gb, x1, y1-1, test_np) # Gradient Boost
print("svm classifier...")
svc_oof_train, svc_oof_test = get_oof(svc,x1, y1-1, test_np) # Support Vector Classifier

def getLogLoss(classifier,data,y_true):
    y_onehot = np_utils.to_categorical(np.array(y_true))
    y_pred = classifier.predict_proba(data)
    return sklearn.metrics.log_loss(y_onehot,y_pred)

print("random forest log-loss : {:.4f}".format(getLogLoss(rf,x2,y2-1)))
print("extra tree log-loss : {:.4f}".format(getLogLoss(et,x2,y2-1)))
print("adaboost log-loss : {:.4f}".format(getLogLoss(ada,x2,y2-1)))
print("gradient boost log-loss : {:.4f}".format(getLogLoss(gb,x2,y2-1)))
print("svm log-loss : {:.4f}".format(getLogLoss(svc,x2,y2-1)))
# ensemble
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
gbm = xgb.XGBClassifier(
    learning_rate = 0.01,
 n_estimators= 2000,
 max_depth= 5,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softprob',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y1-1)
predictions = gbm.predict_proba(x_test)
submission_ens = pd.DataFrame(predictions, columns=['class'+str(c+1) for c in range(9)])
submission_ens['ID'] = test_var["ID"]
submission_ens.to_csv('submission_ensemble_tfidf_withGeneFeatures.csv', index=False)