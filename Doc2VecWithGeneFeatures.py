#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 18:57:19 2017

@author: suresh
"""

import pandas as pd
import numpy as np
from sklearn import *
from gensim.models import Doc2Vec
from sklearn.cross_validation import train_test_split
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

train_var = pd.read_csv('data/training_variants')
test_var = pd.read_csv('data/test_variants')
train_text = pd.read_csv('data/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv('data/test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
gene_class = pd.read_csv('data/Census_all.csv',sep=",")
gene_colnames = ['Gene Symbol','Hallmark','Molecular Genetics','Role in Cancer','Mutation Types']
gene_class_trim = gene_class[gene_colnames]
#train = pd.merge(train_var, train_text, how='left', on='ID').fillna('')
#y = train['Class'].values
##train = train.drop(['Class'], axis=1)
#
#test = pd.merge(test_var, test_text, how='left', on='ID').fillna('')
#pid = test['ID'].values
#train_var["HallMarkGene"] = train_var.apply(lambda r: 1 if gene_class_trim.loc[(gene_class_trim["Gene Symbol"]==r['Gene']).any() ]["Hallmark"]=="Yes" else 0, axis=1)
train_var_withFeatures = pd.merge(train_var,gene_class_trim, how="left",left_on="Gene",right_on="Gene Symbol")

train_var_withFeatures["Hallmark"] = np.where(train_var_withFeatures['Hallmark']=='Yes', 1, 0)
train_var_withFeatures["Gene_Trans"] = np.where(train_var_withFeatures['Mutation Types'].str.contains('T'), 1, 0)
train_var_withFeatures["Gene_Missense"] = np.where(train_var_withFeatures['Mutation Types'].str.contains('Mis'), 1, 0)
train_var_withFeatures["Gene_Nonsense"] = np.where(train_var_withFeatures['Mutation Types'].str.contains('N'), 1, 0)
train_var_withFeatures["Gene_Frameshift"] = np.where(train_var_withFeatures['Mutation Types'].str.contains('F'), 1, 0)
train_var_withFeatures["Gene_Amplification"] = np.where(train_var_withFeatures['Mutation Types'].str.contains('A'), 1, 0)
train_var_withFeatures["Gene_Deletion"] = np.where(train_var_withFeatures['Mutation Types'].str.contains('D'), 1, 0)
train_var_withFeatures["Gene_Splicesite"] = np.where(train_var_withFeatures['Mutation Types'].str.contains('S'), 1, 0)
train_var_withFeatures["Gene_Promoter"] = np.where(train_var_withFeatures['Mutation Types'].str.contains('Promoter'), 1, 0)
train_var_withFeatures["Gene_Other"] = np.where(train_var_withFeatures['Mutation Types'].str.contains('O'), 1, 0)
train_var_withFeatures["Oncogene"] = np.where(train_var_withFeatures['Role in Cancer'].str.contains('oncogene'), 1, 0)
train_var_withFeatures["Fusion"] = np.where(train_var_withFeatures['Role in Cancer'].str.contains('fusion'), 1, 0)
train_var_withFeatures["TSG"] = np.where(train_var_withFeatures['Role in Cancer'].str.contains('TSG'), 1, 0)
train_var_withFeatures["Dominant"] = np.where(train_var_withFeatures['Molecular Genetics'].str.contains('Dom'), 1, 0)
train_var_withFeatures["Recessive"] = np.where(train_var_withFeatures['Molecular Genetics'].str.contains('Rec'), 1, 0)
train_var_withFeatures["XLinked"] = np.where(train_var_withFeatures['Molecular Genetics'].str.contains('X'), 1, 0)
train_var_withFeatures['Variation_First'] = train_var_withFeatures.apply(lambda r: r['Variation'].lower()[0], axis=1)
train_var_withFeatures['Variation_Last'] = train_var_withFeatures.apply(lambda r : r['Variation'].lower()[len(r['Variation'].lower())-1],axis=1)
train_var_withFeatures['AAChange'] = train_var_withFeatures['Variation_First']+train_var_withFeatures['Variation_Last']

##prepare test data
test_var_withFeatures = pd.merge(test_var,gene_class_trim, how="left",left_on="Gene",right_on="Gene Symbol")

test_var_withFeatures["Hallmark"] = np.where(test_var_withFeatures['Hallmark']=='Yes', 1, 0)
test_var_withFeatures["Gene_Trans"] = np.where(test_var_withFeatures['Mutation Types'].str.contains('T'), 1, 0)
test_var_withFeatures["Gene_Missense"] = np.where(test_var_withFeatures['Mutation Types'].str.contains('Mis'), 1, 0)
test_var_withFeatures["Gene_Nonsense"] = np.where(test_var_withFeatures['Mutation Types'].str.contains('N'), 1, 0)
test_var_withFeatures["Gene_Frameshift"] = np.where(test_var_withFeatures['Mutation Types'].str.contains('F'), 1, 0)
test_var_withFeatures["Gene_Amplification"] = np.where(test_var_withFeatures['Mutation Types'].str.contains('A'), 1, 0)
test_var_withFeatures["Gene_Deletion"] = np.where(test_var_withFeatures['Mutation Types'].str.contains('D'), 1, 0)
test_var_withFeatures["Gene_Splicesite"] = np.where(test_var_withFeatures['Mutation Types'].str.contains('S'), 1, 0)
test_var_withFeatures["Gene_Promoter"] = np.where(test_var_withFeatures['Mutation Types'].str.contains('Promoter'), 1, 0)
test_var_withFeatures["Gene_Other"] = np.where(test_var_withFeatures['Mutation Types'].str.contains('O'), 1, 0)
test_var_withFeatures["Oncogene"] = np.where(test_var_withFeatures['Role in Cancer'].str.contains('oncogene'), 1, 0)
test_var_withFeatures["Fusion"] = np.where(test_var_withFeatures['Role in Cancer'].str.contains('fusion'), 1, 0)
test_var_withFeatures["TSG"] = np.where(test_var_withFeatures['Role in Cancer'].str.contains('TSG'), 1, 0)
test_var_withFeatures["Dominant"] = np.where(test_var_withFeatures['Molecular Genetics'].str.contains('Dom'), 1, 0)
test_var_withFeatures["Recessive"] = np.where(test_var_withFeatures['Molecular Genetics'].str.contains('Rec'), 1, 0)
test_var_withFeatures["XLinked"] = np.where(test_var_withFeatures['Molecular Genetics'].str.contains('X'), 1, 0)
test_var_withFeatures['Variation_First'] = test_var_withFeatures.apply(lambda r: r['Variation'].lower()[0], axis=1)
test_var_withFeatures['Variation_Last'] = test_var_withFeatures.apply(lambda r : r['Variation'].lower()[len(r['Variation'].lower())-1],axis=1)
test_var_withFeatures['AAChange'] = test_var_withFeatures['Variation_First']+train_var_withFeatures['Variation_Last']

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
train_var_withFeatures["FirstAliphatic"] = train_var_withFeatures['Variation_First'].apply(isAliphatic)
train_var_withFeatures["LastAliphatic"] = train_var_withFeatures['Variation_Last'].apply(isAliphatic)
train_var_withFeatures["FirstAromatic"] = train_var_withFeatures['Variation_First'].apply(isAromatic)
train_var_withFeatures["LastAromatic"] = train_var_withFeatures['Variation_Last'].apply(isAromatic)
train_var_withFeatures["FirstNeutral"] = train_var_withFeatures['Variation_First'].apply(isNeutral)
train_var_withFeatures["LastNeutral"] = train_var_withFeatures['Variation_Last'].apply(isNeutral)
train_var_withFeatures["FirstAcidic"] = train_var_withFeatures['Variation_First'].apply(isAcidic)
train_var_withFeatures["LastAcidic"] = train_var_withFeatures['Variation_Last'].apply(isAcidic)
train_var_withFeatures["FirstBasic"] = train_var_withFeatures['Variation_First'].apply(isBasic)
train_var_withFeatures["LastBasic"] = train_var_withFeatures['Variation_Last'].apply(isBasic)
train_var_withFeatures["FirstUnique"] = train_var_withFeatures['Variation_First'].apply(isUnique)
train_var_withFeatures["LastUnique"] = train_var_withFeatures['Variation_Last'].apply(isUnique)
train_var_withFeatures["FirstCharged"] = train_var_withFeatures['Variation_First'].apply(isCharged)
train_var_withFeatures["LastCharged"] = train_var_withFeatures['Variation_Last'].apply(isCharged)
train_var_withFeatures["FirstPolar"] = train_var_withFeatures['Variation_First'].apply(isPolar)
train_var_withFeatures["LastPolar"] = train_var_withFeatures['Variation_Last'].apply(isPolar)
train_var_withFeatures["FirstHydrophobic"] = train_var_withFeatures['Variation_First'].apply(isHydrophobic)
train_var_withFeatures["LastHydrophobic"] = train_var_withFeatures['Variation_Last'].apply(isHydrophobic)

##test features
test_var_withFeatures["FirstAliphatic"] = test_var_withFeatures['Variation_First'].apply(isAliphatic)
test_var_withFeatures["LastAliphatic"] = test_var_withFeatures['Variation_Last'].apply(isAliphatic)
test_var_withFeatures["FirstAromatic"] = test_var_withFeatures['Variation_First'].apply(isAromatic)
test_var_withFeatures["LastAromatic"] = test_var_withFeatures['Variation_Last'].apply(isAromatic)
test_var_withFeatures["FirstNeutral"] = test_var_withFeatures['Variation_First'].apply(isNeutral)
test_var_withFeatures["LastNeutral"] = test_var_withFeatures['Variation_Last'].apply(isNeutral)
test_var_withFeatures["FirstAcidic"] = test_var_withFeatures['Variation_First'].apply(isAcidic)
test_var_withFeatures["LastAcidic"] = test_var_withFeatures['Variation_Last'].apply(isAcidic)
test_var_withFeatures["FirstBasic"] = test_var_withFeatures['Variation_First'].apply(isBasic)
test_var_withFeatures["LastBasic"] = test_var_withFeatures['Variation_Last'].apply(isBasic)
test_var_withFeatures["FirstUnique"] = test_var_withFeatures['Variation_First'].apply(isUnique)
test_var_withFeatures["LastUnique"] = test_var_withFeatures['Variation_Last'].apply(isUnique)
test_var_withFeatures["FirstCharged"] = test_var_withFeatures['Variation_First'].apply(isCharged)
test_var_withFeatures["LastCharged"] = test_var_withFeatures['Variation_Last'].apply(isCharged)
test_var_withFeatures["FirstPolar"] = test_var_withFeatures['Variation_First'].apply(isPolar)
test_var_withFeatures["LastPolar"] = test_var_withFeatures['Variation_Last'].apply(isPolar)
test_var_withFeatures["FirstHydrophobic"] = test_var_withFeatures['Variation_First'].apply(isHydrophobic)
test_var_withFeatures["LastHydrophobic"] = test_var_withFeatures['Variation_Last'].apply(isHydrophobic)

aachangepd = pd.get_dummies(train_var_withFeatures['AAChange'])
aachangepdTest = pd.get_dummies(test_var_withFeatures['AAChange'])
train_var_withFeatures = pd.concat((train_var_withFeatures,aachangepd),axis=1)
test_var_withFeatures = pd.concat((test_var_withFeatures,aachangepdTest),axis=1)

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
def getDoc2VecFeatures(modelFile,startIndex,endIndex,ncols):
    doc2VecModel= Doc2Vec.load(modelFile)
    train_arrays = np.zeros(((endIndex-startIndex), ncols))
    for i in range(startIndex,endIndex):
        train_arrays[i-startIndex] = doc2VecModel.docvecs['Text_'+str(i)]
    return train_arrays
doc2VecFeatures = getDoc2VecFeatures('./docEmbeddings.d2v',0,train_var.shape[0],100)
trainWithFeaturesForTraining = pd.concat((trainWithFeaturesForTraining,pd.DataFrame(doc2VecFeatures)),axis=1)
doc2VecFeaturesTest = getDoc2VecFeatures('./docEmbeddings.d2v',train_var.shape[0],(train_var.shape[0]+test_var.shape[0]),100)
testWithFeaturesForTesting = pd.concat((testWithFeaturesForTesting,pd.DataFrame(doc2VecFeaturesTest)),axis=1)


def train_xgboost(trainWithFeaturesForTraining,y,fold):
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
def gen_nn(input_dim=512, width=32, depth=4,do=0.2,l1_wt=0.01,l2_wt=0.01):
    model = Sequential()
    model.add(Dropout(do,input_shape=(input_dim,)))
    #model.add(Dense(input_dim=input_dim, units=width))
    model.add(Dense(units=width,kernel_constraint=maxnorm(3),kernel_regularizer=l1_l2(l1_wt,l2_wt)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    for k in range(2, depth):
        model.add(Dense(units=width*(k+1),kernel_constraint=maxnorm(3),kernel_regularizer=l1_l2(l1_wt,l2_wt)))
        model.add(Dropout(do,input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
#    model.compile(optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=0.001, decay=1e-6, beta_1=0.999,beta_2=0.99,epsilon=1e-5),
                  #loss='categorical_crossentropy',
                  loss = 'kld',
                  metrics=['accuracy'])
    return model
checkpointer = ModelCheckpoint(filepath='doc2vecWithEngineeredFeatures.hdf5', 
                               verbose=0, save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=15, verbose=0, mode='auto')

def train_nn(trainWithFeaturesForTraining,y):
    if np.min(y) !=0:
        y = y-1
    y_onehot = np_utils.to_categorical(np.array(y))
    X_train, X_test, y_train, y_test = train_test_split(trainWithFeaturesForTraining, y_onehot, test_size=0.25, random_state=0)
    batch_size = 32
    model = gen_nn(input_dim=X_train.shape[1],width=16,depth=4,do=0.2,l1_wt=0.001,l2_wt=0.01)
    his = model.fit(X_train, y_train, nb_epoch=4000, \
                      batch_size=batch_size, \
                      validation_split=0.2, \
                      shuffle=True, callbacks=[checkpointer,earlyStopping],verbose=0)
    train_loss = his.history['loss']
    val_loss = his.history['val_loss']
    print ("min train loss {0}, min val loss {1}".format(min(train_loss),min(val_loss)))
    plt.plot(range(1, len(train_loss)+1), train_loss, color='blue', label='Train loss')
    plt.plot(range(1, len(val_loss)+1), val_loss, color='red', label='Val loss')
    plt.xlim(0, len(train_loss))
    plt.legend(loc="upper right")
    plt.xlabel('#Epoch')
    plt.ylabel('Loss')
    # evaluate trained model
    model.load_weights('doc2vecWithEngineeredFeatures.hdf5')
    score = model.evaluate(X_test, y_test, verbose=0)
    print('\nTest loss: %.3f' % score[0])
    print('Test accuracy: %.3f' % score[1])

#train_nn(np.array(trainWithFeaturesForTraining),train_var['Class'].values)
#train_xgboost(trainWithFeaturesForTraining,train_var['Class'].values,1)
def get_siamese(input_dim=512, width=32, depth=4,do=0.2,l1_wt=0.01,l2_wt=0.01):
    model = Sequential()
    #model.add(Dropout(do,input_shape=(input_dim,)))
    left_input = Input((input_dim,))
    right_input = Input((input_dim,))
    model.add(Dense(input_dim=input_dim, units=width))
    model.add(Dense(units=width,kernel_constraint=maxnorm(3),kernel_regularizer=l1_l2(l1_wt,l2_wt)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    for k in range(2, depth):
        model.add(Dense(units=width*(k+1),kernel_constraint=maxnorm(3),kernel_regularizer=l1_l2(l1_wt,l2_wt)))
        model.add(Dropout(do,input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Dense(units=width*depth,activation="sigmoid",kernel_regularizer=l1_l2(l1_wt,l2_wt)))
    #encode each of the two inputs into a vector with the nn model
    left_encoded = model(left_input)
    right_encoded = model(right_input)
    #merge two encoded inputs with the l1 distance between them
    L1_distance = lambda x: K.abs(x[0]-x[1])
    both = merge([left_encoded,right_encoded], mode = L1_distance, output_shape=lambda x: x[0])
    prediction = Dense(1,activation='sigmoid')(both)
    siamese_net = Model(input=[left_input,right_input],output=prediction) 
    optimizer = Adam(0.00006)
    #siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    return siamese_net
def train_siamese(trainWithFeaturesForTraining,y,iterations):
    if np.min(y) !=0:
        y = y-1
    #y_onehot = np_utils.to_categorical(np.array(y))
    X_train, X_test, y_train, y_test = train_test_split(trainWithFeaturesForTraining, y, test_size=0.25, random_state=0)
    evaluate_every = 500
    loss_every=50
    batch_size = 32
    N_way = 20
    n_val = 250
    loader = Siamese_Loader(X_train,X_test,y_train,y_test,len(np.unique(y)))
    siamese_net = get_siamese(input_dim=X_train.shape[1],width=64,depth=8,do=0,l1_wt=0.00,l2_wt=0.0)
    best = 0.
    train_losses =[]
    val_accs = []
    for i in range(iterations):
        (inputs,targets)=loader.get_batch(batch_size)
        loss=siamese_net.train_on_batch(inputs,targets)
        train_losses.append(loss)
        if i % evaluate_every == 0:
            val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
            val_accs.append(val_acc)
            if val_acc >= best:
                print("saving")
                siamese_net.save('one-shot-weights')
                best=val_acc
    
        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i,loss))
    plt.plot(range(1, len(train_losses)+1), train_losses, color='blue', label='Train loss')
    plt.plot(range(1, len(val_accs)+1), val_accs, color='blue', label='Validation Accuracy')

#train_siamese((trainWithFeaturesForTraining),train_var['Class'].values,50000)

##ensembling
ntrain = trainWithFeaturesForTraining.shape[0]
ntest = testWithFeaturesForTesting.shape[0]
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
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
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
et_oof_train, et_oof_test = get_oof(et, trainWithFeaturesForTraining.values, train_var['Class'].values-1, testWithFeaturesForTesting.values) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,trainWithFeaturesForTraining.values, train_var['Class'].values-1, testWithFeaturesForTesting.values) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, trainWithFeaturesForTraining.values, train_var['Class'].values-1, testWithFeaturesForTesting.values) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,trainWithFeaturesForTraining.values, train_var['Class'].values-1, testWithFeaturesForTesting.values) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,trainWithFeaturesForTraining.values, train_var['Class'].values-1, testWithFeaturesForTesting.values) # Support Vector Classifier

# ensemble
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softprob',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, train_var['Class'].values-1)
predictions = gbm.predict_proba(x_test)
submission_ens = pd.DataFrame(predictions, columns=['class'+str(c+1) for c in range(9)])
submission_ens['ID'] = test_var["ID"]
submission_ens.to_csv('submission_ensemble_doc2vec_withGeneFeatures.csv', index=False)
