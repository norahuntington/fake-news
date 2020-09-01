import numpy as np
import nltk
import pickle

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from clean_data import get_data, split_X_y

def vectorize_train_data(X_train, vec_class):
    '''
        Transforms array of statements into a matrix of weighted word frequencies using TfidfVectorizer class.

        INPUT:
            - Array of statements
            - TfidfVectorizer class
        OUTPUT:
            - Tfidf object
            - Matrix of floats
    '''
    matrix = vec_class.fit_transform(X_train)
    X_train_tfidf = matrix.toarray()
    return vec_class, X_train_tfidf

def vectorize_test_data(X_test, trans_vec):
    '''
        Transforms array of statements into a matrix using a fitted TfidfVector.

        INPUT:
            - Array of statements
            - Fitted TfidfVector
        OUTPUT:
            - Matrix of floats
    '''
    matrix = trans_vec.transform(X_test)
    X_test_tfidf = matrix.toarray()
    return X_test_tfidf

def stemmer(text):
    '''
        Transforms words in a text to their stems.
        Allows TfidfVector class to use the SnowballStemmer.

        INPUT:
            - String of text
        OUTPUT:
            - List of stemmed words
    '''
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(SnowballStemmer('english').stem(item))
    return stems

def lemmatizer(text):
    '''
        Transforms words in a text to their meaningful base.
        Allows TfidfVector class to use the WordNetLammatizer.

        INPUT:
            - String of text
        OUTPUT:
            - List of words bases
    '''
    tokens = nltk.word_tokenize(text)
    lems = []
    for item in tokens:
        lems.append(WordNetLemmatizer().lemmatize(item))
    return lems

def cross_val_auc_roc(model, tfidf, X, y, folds = 5):
    '''
        Does cross validation on given X and y training data.
        Uses area under ROC curve to score models- returns mean of folds number of runs.

        INPUT:
            - Model class
            - TfidfVectorizer class
            - Array X
            - Array y
            - Int folds
        OUTPUT:
            - Float
    '''
    kf = KFold(n_splits=folds)
    auc_roc = []
    for train, test in kf.split(X):
        vec_trans, X_train_tfidf = vectorize_train_data(X[train], tfidf)
        X_test_tfidf = vectorize_test_data(X[test], vec_trans)
        model.fit(X_train_tfidf, y[train])
        y_predict = model.predict_proba(X_test_tfidf)
        auc_roc.append(roc_auc_score(y[test], y_predict[:,1]))
    return np.mean(auc_roc)

def score_final_model(model, tfidf, X_train, y_train, X_test, y_test):
    '''
        Transforms X_train into a Tfidf matrix. Transforms X_test into matrix using fitted Tfidf class.
        Uses X_train_tfidf matrix and y_train to fit model.
        Scores model using area under ROC curve.

        INPUT:
            - Model class
            - TfidfVectorizer class
            - Array X_train
            - Array y_train
            - Array X_test
            - Array y_test
        OUTPUT:
            - Float
    '''
    vec_trans, X_train_tfidf = vectorize_train_data(X_train, tfidf)
    X_test_tfidf = vectorize_test_data(X_test, vec_trans)
    model.fit(X_train_tfidf, y_train)
    y_predict = model.predict_proba(X_test_tfidf)
    return roc_auc_score(y_test, y_predict[:,1])


if __name__ == '__main__':
  #Prep data for model:
  data = get_data()
  X, y = split_X_y(data)
  cutoff = round(len(X)*.8)
  X_train, X_test, y_train, y_test = X[:cutoff], X[cutoff:], y[:cutoff], y[cutoff:]

  #Some possible TfidfVectorizer()s:
  #tfidf_base = TfidfVectorizer(strip_accents='ascii')
  #tfidf_stopwords = TfidfVectorizer(strip_accents='ascii', stop_words='english')
  #tfidf_stemmer = TfidfVectorizer(strip_accents='ascii', tokenizer=stemmer)
  #tfidf_lemmatizer = TfidfVectorizer(strip_accents='ascii', tokenizer=lemmatizer)
  #tfidf_limit_features = TfidfVectorizer(strip_accents='ascii', max_features=5000)
  #tfidf_ngrams = TfidfVectorizer(strip_accents='ascii', ngram_range=(1,3), max_features=15000)

  #How to score a model:
  #rf_model = RandomForestClassifier()
  #rf_tfidf = TfidfVectorizer(strip_accents='ascii')
  #rf_score = cross_val_auc_roc(rf_model, rf_tfidf, X_train, y_train, folds=5)
  #print(rf_score)

  #How to score final model using X_test, y_test data:
  model = RandomForestClassifier()
  tfidf = TfidfVectorizer(strip_accents='ascii')
  model_score = score_final_model(model, tfidf, X_train, y_train, X_test, y_test)
  print(model_score)
  pickle.dump(model, open('Models/rf_model.pkl', 'wb'))


  #To load saved file:
  #filename = 'Models/rf_model.pkl'
  #model = pickle.load(open(filename, 'rb'))