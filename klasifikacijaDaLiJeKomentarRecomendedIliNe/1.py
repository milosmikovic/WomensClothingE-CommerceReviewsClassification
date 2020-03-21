from __future__ import unicode_literals
import numpy as np 
import pandas as pd 
import nltk
import re
import string
import sklearn.metrics as met

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

def prep_clean(text):
    text = text.lower()
    text = re.sub(r'\d+','',text)
    tokens = text.split(" ")
    words = [token for token in tokens if not token in stop_words]
    words = [stemmer.stem(word) for word in words]
    words = [word for word in words if not word in string.punctuation]
    words = [word for word in words if len(word) > 1]
    return words

def pipeline(bow, tfidf, model):
    return Pipeline([('bow', bow),
               ('tfidf', tfidf),
               ('classifier', model),
              ])

def model(mod, name, X_train, X_test, y_train, y_test):
    mod.fit(X_train, y_train)
    print(name)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv=5)
    predictions = cross_val_predict(mod, X_train, y_train, cv=5)
    print("Accuracy: ", round(acc.mean(),3))
    cm = confusion_matrix(predictions, y_train)
    print("Confusion Matrix: \n", cm)
    print("Classification Report: \n", classification_report(predictions, y_train))
    plt.imshow(cm)
    plt.colorbar()
    plt.title('Matrica konfuzije')
    plt.xticks(range(2), ['0','1'])
    plt.yticks(range(2), ['0','1'])
    plt.show()

    print("--------")
    print("TEST MATRICA KONF")

    y_predicted = mod.predict(X_test)

    print(met.accuracy_score(y_test, y_predicted))
    mat_konf = met.confusion_matrix(y_test, y_predicted)
    print(mat_konf)

    plt.imshow(mat_konf)
    plt.colorbar()
    plt.title('Matrica konfuzije')
    plt.xticks(range(2), ['0','1'])
    plt.yticks(range(2), ['0','1'])
    plt.show()

# PODACI
df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv',index_col = 0)
df.info()
df.groupby('Recommended IND').describe()



# PREDPROCESIRANJE

# koliko ima null rew text pre
print("BROJ NULL VREDNOSTI:")
print(df.isnull().sum())
print("###################################")
df = df.dropna(subset=['Review Text'])
# posle dropna
print("BROJ NULL VREDNOSTI POSLE CISCENJA:")
print(df.isnull().sum()) 
print("###################################")
rec = df.filter(['Review Text','Recommended IND'])
rec.columns = ['text','target']

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

X_train, X_test, y_train, y_test = train_test_split(rec['text'], rec['target'], test_size=0.2)

mnb = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), MultinomialNB())
mnb = model(mnb, "Multinomial Naive Bayes", X_train, X_test, y_train, y_test)


svc = pipeline(CountVectorizer(analyzer=prep_clean, ngram_range=(1,2)), TfidfTransformer(), LinearSVC())
svc = model(svc, "Linear SVC", X_train, X_test, y_train, y_test)