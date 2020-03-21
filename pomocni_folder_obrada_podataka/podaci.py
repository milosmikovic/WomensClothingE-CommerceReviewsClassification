from __future__ import unicode_literals
import numpy as np 
import pandas as pd 
import nltk
import re
import string

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
import collections
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

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
df = df.dropna()

corpus = []
comments = df['Review Text']

# tmp_df = df.filter(['Review Text','Recommended IND'])
# tmp_df.columns = ['text','target']
# tmp_df1 = df.loc[df['Recommended IND'] == 1]
# pozitivni = tmp_df1['Review Text']
# tmp_df2 = df.loc[df['Recommended IND'] == 0]
# negativni = tmp_df2['Review Text']

# print("#############")
# print(len(pozitivni))
# print(len(negativni))
# print("#############")

for comment in comments:
    processed_comment = prep_clean(comment)
    corpus.append(processed_comment)


count = 0

with open('preproc_komentari.txt', 'w') as f:
    for line in corpus:
        for word in line:
            count += 1
            f.write("%s " % word)
            if(count == 10):
                f.write("\n")
                count = 0

# df.to_csv (r'C:\Users\Ron\Desktop\export_dataframe.csv', index = None, header=True)

# wordcount = {}
# for line in corpus:
#     for word in line:
#         if word not in wordcount:
#             wordcount[str(word)] = 1
#         else:
#             wordcount[str(word)] += 1


# broj_reci = int(input("Koliko reci zelite da ispisete pozitivnih: "))
# word_counter = collections.Counter(wordcount)
# for word, count in word_counter.most_common(broj_reci):
#     print(word, ": ", count)


# corpus1 = []
# for comment in negativni:
#     processed_comment = prep_clean(comment)
#     corpus1.append(processed_comment)
#     # print(processed_comment)

# wordcount1 = {}
# for line in corpus1:
#     for word in line:
#         if word not in wordcount1:
#             wordcount1[str(word)] = 1
#         else:
#             wordcount1[str(word)] += 1


# broj_reci = int(input("Koliko reci zelite da ispisete negativnih: "))
# word_counter1 = collections.Counter(wordcount1)
# for word, count in word_counter1.most_common(broj_reci):
#     print(word, ": ", count)

# # tmp_df = df.loc[df['Clothing ID'] == c_id]