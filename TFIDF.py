# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import *
import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.callbacks import ModelCheckpoint 
import matplotlib.pyplot as plt
import re
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
def gen_nn(input_dim=512, width=32, depth=4):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, units=width))
    model.add(Activation('relu'))
    for k in range(2, depth):
        model.add(Dense(units=width*(k+1)))
        model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    model.compile(optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
nltk.download('stopwords')
stop = stopwords.words('english')
def tokenizer_stem_nostop(text):
    porter = PorterStemmer()
    return [porter.stem(w) for w in re.split('\s+', text.strip()) \
            if w not in stop and re.match('[a-zA-Z]+', w)]
def preprocess(text):
    text = text.lower()
    text.translate(str.maketrans('','',string.punctuation))
    return text
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),stop_words='english', sublinear_tf=True, 
                             preprocessor=preprocess, tokenizer=tokenizer_stem_nostop)
X = vectorizer.fit_transform(train["Text"]).toarray()
y_onehot = np_utils.to_categorical(np.array(y))
# the dictionary map from word to feature index
dictionary = vectorizer.vocabulary_

# construct inverse_dictionary for later use
inverse_dictionary = {v: k for k, v in dictionary.items()}
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.25, random_state=0)
batch_size = 32

model = gen_nn(input_dim=X_train.shape[1])
checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)
his = model.fit(X_train, y_train, nb_epoch=400, \
                      batch_size=batch_size, \
                      validation_split=0.2, \
                      shuffle=True, callbacks=[checkpointer],verbose=1)
train_loss = his.history['loss']
val_loss = his.history['val_loss']
plt.plot(range(1, len(train_loss)+1), train_loss, color='blue', label='Train loss')
plt.plot(range(1, len(val_loss)+1), val_loss, color='red', label='Val loss')
plt.xlim(0, len(train_loss))
plt.legend(loc="upper right")
plt.xlabel('#Epoch')
plt.ylabel('Loss')
# evaluate trained model
model.load_weights('weights.best.from_scratch.hdf5')
score = model.evaluate(X_test, y_test, verbose=0)
print('\nTest loss: %.3f' % score[0])
print('Test accuracy: %.3f' % score[1])
test_data = vectorizer.transform(test["Text"])
test_prediction = model.predict(test_data.toarray())
submission = pd.DataFrame(test_prediction, columns=['class'+str(c) for c in range(10)])
del submission["class0"]
submission['ID']=pid
cols = submission.columns.tolist()
cols = cols[-1:] + cols[:-1]
submission = submission[cols]
submission.to_csv('tfidf.csv', index=False)
