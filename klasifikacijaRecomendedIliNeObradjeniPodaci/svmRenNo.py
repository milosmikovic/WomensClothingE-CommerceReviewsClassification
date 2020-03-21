from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.naive_bayes import  MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import nltk

import sklearn.metrics as met

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import sklearn.preprocessing as prep

import matplotlib.pyplot as plt

porter = PorterStemmer()
lancaster=LancasterStemmer()

def stemSentence(sentence):

    if type(sentence) == float : 
      print(sentence)
      return

    token_words=sentence.split(" ")
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')

df = df.dropna()

comments = df['Review Text']
classes = df['Recommended IND']

print(comments.shape)
print(classes.shape)

corpus = []
count = 0

for comment in comments :
    processed_comment = stemSentence(comment)
    corpus.append(processed_comment)
    count+= 1

vectorizer = TfidfVectorizer(stop_words='english')
x_fitted = vectorizer.fit_transform(corpus)

x_train, x_test, y_train, y_test = train_test_split(x_fitted, classes, stratify = classes, test_size = 0.3)
clf = MultinomialNB()
clf.fit(x_train, y_train)
df_visualisator = pd.DataFrame(clf.feature_count_, index =clf.classes_,  columns = vectorizer.get_feature_names())

products_id = df['Clothing ID'].unique()
new_df = []
print(df_visualisator)

for c_id in products_id :
    diff_of_com = 0
    sum_of_words = 0
    tmp_df = df.loc[df['Clothing ID'] == c_id]
    avg_rating = np.average(tmp_df['Rating'])
    avg_rating = round(avg_rating)
    avg_feedback = np.average(tmp_df['Positive Feedback Count'])
    avg_age = np.average(tmp_df['Age'])
    num_reviews = tmp_df.shape[0]

    for comment in tmp_df['Review Text'] :
        sum_of_words += len(comment)
        x_predict = vectorizer.transform([comment])
        y_predict = clf.predict(x_predict)
        probabilies = clf.predict_proba(x_predict)
        s = pd.Series(probabilies[0], index = clf.classes_)
        diff_of_com += s[1] - s[0]

    avg_word_count = sum_of_words*1.0/num_reviews
    new_df.append([c_id, avg_rating, num_reviews, avg_word_count, diff_of_com / num_reviews, avg_feedback, avg_age])

made_df = pd.DataFrame(new_df, columns = ['ID', 'Avg_Rating', 'Num_Reviews', 'AVG_cnt', 'Pos_Negative_Proba','Avg_Feedback', 'Avg_age'])

print(made_df)

features = made_df.columns[3:].tolist()
x=made_df[features]
y = made_df['Avg_Rating']

# recomended ili ne
y = [0 if e < 3 else 1 for e in y]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# parameters_for_SVC = [{'C': [pow(2,x) for x in range(-6,10,2)],
#                'kernel' : ['linear']
#                },

#               {'C': [pow(2,x) for x in range(-6,10,2)],
#                'kernel': ['poly'],
#                'degree': [2, 3, 4, 5],
#                'gamma': np.arange(0.1, 1.1, 0.1),
#                'coef0': np.arange(0, 2, 0.5)
#               }
#                ,
#                  {'C': [pow(2,x) for x in range(-6,10,2)],
#                'kernel' : ['rbf'],
#                'gamma': np.arange(0.1, 1.1, 0.1)
#                },
#                {'C': [pow(2,x) for x in range(-6,10,2)],
#                'kernel' : ['sigmoid'],
#                'gamma': np.arange(0.1, 1.1, 0.1),
#                'coef0': np.arange(0, 2, 0.5)
#                }]
# clf = GridSearchCV(SVC(), parameters_for_SVC, cv=5, scoring='precision_macro')
clf = SVC()
clf.fit(x_train, y_train)

print("##################")

y_predicted = clf.predict(x_test)

print(met.accuracy_score(y_test, y_predicted))
mat_konf = met.confusion_matrix(y_test, y_predicted)
print(mat_konf)

plt.imshow(mat_konf)
plt.colorbar()
plt.title('Matrica konfuzije')
plt.xticks(range(2), ['0','1'])
plt.yticks(range(2), ['0','1'])
plt.show()



y_predicted = clf.predict(x_train)
print("TRENING PODACI")
print(met.accuracy_score(y_train, y_predicted))
mat_konf = met.confusion_matrix(y_train, y_predicted)
print(mat_konf)

plt.imshow(mat_konf)
plt.colorbar()
plt.title('Matrica konfuzije')
plt.xticks(range(2), ['0','1'])
plt.yticks(range(2), ['0','1'])
plt.show()