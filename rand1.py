import nltk
nltk.data.path.append('./nltk_data/')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import preprocessing
Encode = preprocessing.LabelEncoder()
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
    path = "./008_official.txt"
    df_test = []

    with open(path) as fp:
        for line1 in fp:
            df_test.append((line1))
    df_orginal = df_test

    print('Original Input :', df_test[3])
    df_test = [preprocess(x) for x in df_test]
    print('Pre Processing based :',df_test[3])
    df_test = [txtblb(x) for x in df_test]
    print('Parts of speech based:',df_test[3])

    # Loading Trained Model -- Multilayer Preceptor
    clf = joblib.load('NN100Lyr.pkl')
    predicted = clf.predict(df_test)
    print("Our Prediction :" , predicted[3])



if __name__ == "__main__":
    main()
