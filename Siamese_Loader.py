#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 11:44:42 2017

@author: suresh
"""
import numpy as np
import numpy.random as rng
from sklearn.utils import shuffle
class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""    
    def __init__(self,Xtrain,Xval,Ytrain,Yval,n_classes):
        self.data = {}
        self.categories = {}
        self.data["train"]=np.array(Xtrain)
        self.categories["train"]=Ytrain
        self.data["val"]=np.array(Xval)
        self.categories["val"]=Yval
        self.n_classes = n_classes
        self.ndim = len(Xtrain.columns)
        self.YtrainClassBins = np.bincount(Yval)
    
    def get_batch(self,n,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
        Y=self.categories[s]
        categories = rng.choice(self.n_classes,size=(n,),replace=True)
        pairs=[np.zeros((n, self.ndim)) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            category = categories[i]
            idx_1 = rng.randint(0,self.YtrainClassBins[category])
            pairs[0][i,:] = X[Y==category][idx_1]#choose a random index from the subset containing just this category's data
            #idx_2 = rng.randint(0,self.n_examples)
            #pick images of same class for 1st half, different for 2nd
            category_2 = category if i >= n//2 else (category + rng.randint(1,self.n_classes)) % self.n_classes
            idx_2 = rng.randint(0,self.YtrainClassBins[category_2])
            pairs[1][i,:] = X[Y==category_2][idx_2]
        return pairs, targets

    def make_oneshot_task(self,N,s="val"):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        Y=self.categories[s]
        #n_examples = len(Y)
        #categories = rng.choice(range(self.n_classes),size=(N,),replace=False)
        #start_idx, end_idx =self.categories[s][language]
        true_category = rng.randint(0,self.n_classes)
        ex1, ex2 = rng.choice(X[Y==true_category].shape[0],replace=False,size=(2,))
        #test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N,self.w,self.h,1)
        #test_image = np.asarray(X[Y==true_category][ex1,:]*N).reshape(N,self.ndim)# create n copies of true category
        test_image = np.vstack([X[Y==true_category][ex1]]*N)
        indices = rng.randint(0,len(X[Y!=true_category]),size=(N,))
        support_set = X[Y!=true_category][indices,:]
        support_set[0,:] = X[Y==true_category][ex2,:]
        #support_set = support_set.reshape(N,self.w,self.h,1)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image,support_set]

        return pairs, targets

    def test_oneshot(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct