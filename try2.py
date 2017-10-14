import nltk
nltk.data.path.append('./nltk_data/')


import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

import string
import matplotlib.pyplot as plt
import numpy as np

from nltk.corpus import stopwords


from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
#
# import scipy.sparse as sparse
from sklearn.cross_validation import train_test_split
# from sklearn.metrics import confusion_matrix
# from sklearn import metrics
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC, NuSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import MultinomialNB
#
# import seaborn as sns

from textblob import TextBlob
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)


import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

def getsum(item):
    tokens = item.split(' ')
    phr_sum = np.zeros(300, np.float32)

    for token in tokens:
        if token in model:
            phr_sum += model[token]

    return phr_sum


exclude = {'/', '"', '.', '&', '-', '#', '(', ')', '$', '@', '%', '!', '‼️‼️', '‼️', '}', '[', ':', ']', '_', '|', ';',
           '{', '?', '=', '\\', '~', '*', ',', '^', '`', '!'}
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
tagCheck = ['NN','JJ', 'RB', 'VB' ,'NNP','NNS','RBR','RBS','RP']



def preprocess(tweet):
    tweet = tweet.lower()
    tweet = ''.join([i.lower() for i in tweet if i not in exclude])
    tweet = ''.join([i for i in tweet if not i.isdigit()])
    tweet = ' '.join( [w for w in tweet.split() if len(w)>1] )
    tweet = " ".join(x.lower() for x in tweet.split() if x not in stop)
    tweet = " ".join(lemmatizer.lemmatize(x, 'v') for x in tweet.split())
    tweet = " ".join(lemmatizer.lemmatize(x) for x in tweet.split())
    tweet = " ".join(x.lower() for x in word_tokenize(tweet) if x != "'s")
    tweet = " ".join(x.lower() for x in word_tokenize(tweet) if x != "'")
    return tweet

def txtblb(line1):
    l = ''
    blob = TextBlob(line1)
    line1 = blob.tags
    for x in line1:
        if x[1] in tagCheck:
            l += " "+x[0]
    return l



def main():
    path = "./training.xlsx"
    df = pd.read_excel(path, sheetname='Sheet1')

    path_test = 'testdata.xlsx'
    df_test = pd.read_excel(path_test, sheetname='Sheet2')

    # for x,y in zip(df['Column2'],df['Column1']):
    #     k = preprocess(x)
    #     fdist = FreqDist(word.lower() for word in word_tokenize(k))


    df['Column2'] = [preprocess(x) for x in df['Column2']]
    df['Column2'] = [txtblb(x) for x in df['Column2']]

    sentence_vec = []
    for line in df['Column2']:
        sentence_vec.append(getsum(line))
    df['Column2'] = sentence_vec

    x_train,x_test,y_train,y_test = train_test_split(df["Column2"],df['Column1'],random_state=1)

    y_t = (df['Column1'])
    x_t = (df['Column2'])

    model_lstm = Sequential()
    model_lstm.add(LSTM(100, input_shape=300))
    model_lstm.add(Dense(1, activation='sigmoid'))
    model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model_lstm.summary())
    model_lstm.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)



    #Saving model
    joblib.dump(model_lstm, 'NN100Lyr.pkl')

    df_test['Column2'] = [preprocess(x) for x in df_test['Column2']]
    df_test['Column2'] = [txtblb(x) for x in df_test['Column2']]
    sentence_vec = []
    for line in df_test['Column2']:
        sentence_vec.append(getsum(line))
    df_test['Column2'] = sentence_vec


    # Final evaluation of the model
    scores = model_lstm.evaluate(df_test['Column2'], df_test['Column1'], verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


    clf = joblib.load('NN100Lyr.pkl')
    predicted = clf.predict(df_test['Column2'])
    print(accuracy_score(df_test['Column1'], predicted))

    k = 0
    print("completed")


if __name__ == "__main__":
    main()
