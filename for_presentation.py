import nltk
nltk.data.path.append('./nltk_data/')


import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()

import pandas as pd


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
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import confusion_matrix
# from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer


from textblob import TextBlob
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib


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

    path_test = './testdata.xlsx'
    df_test = pd.read_excel(path_test, sheetname='Sheet2')

    print(df['Column2'][1])
    df['Column2'] = [preprocess(x) for x in df['Column2']]
    print(df['Column2'][1])
    df['Column2'] = [txtblb(x) for x in df['Column2']]


    clf = MLPClassifier(max_iter=50)
    y_t = (df['Column1'])
    x_t = (df['Column2'])

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), ('tfidf', TfidfTransformer()),
                         ('clf', clf), ])

    text_clf.fit(x_t, y_t,verbose=True)

    #Saving model
    joblib.dump(text_clf, 'NN100Lyr.pkl')

    df_test['Column2'] = [preprocess(x) for x in df_test['Column2']]
    df_test['Column2'] = [txtblb(x) for x in df_test['Column2']]


    predicted = text_clf.predict(df_test['Column2'])
    print(accuracy_score(df_test['Column1'], predicted))

    clf = joblib.load('NN100Lyr.pkl')
    predicted = clf.predict(df_test['Column2'])
    print(accuracy_score(df_test['Column1'], predicted))

    print("completed")


if __name__ == "__main__":
    main()
