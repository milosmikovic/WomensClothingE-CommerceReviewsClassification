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
    # ako zakomentarisemo narednu liniju klas se vrsi nad svi podacima u suprotnom 
    # one sa manje od 5 reviewa odbacujemo
    if(num_reviews > 5):
        new_df.append([c_id, avg_rating, num_reviews, avg_word_count, diff_of_com / num_reviews, avg_feedback, avg_age])
made_df = pd.DataFrame(new_df, columns = ['ID', 'Avg_Rating', 'Num_Reviews', 'AVG_cnt', 'Pos_Negative_Proba','Avg_Feedback', 'Avg_age'])

print(made_df)
# izdvajanje podataka u csv dokument
# made_df.to_csv (r'preproc_data.csv', index = None, header=True)
# print(len(made_df))

features = made_df.columns[3:].tolist()
x=made_df[features]
y = made_df['Avg_Rating']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

parameters_for_KNN = [
                {'n_neighbors': [3, 4, 5],
               'weights' : ['distance', 'uniform'], 
                'p' : [1, 2]
               }]
0.2578611747273271

clf = GridSearchCV(KNeighborsClassifier(), parameters_for_KNN, cv=5, scoring='precision_macro')
clf.fit(x_train, y_train)

print("##################")
print(clf.best_score_)
print(clf.best_params_)


y_predicted = clf.predict(x_test)

print("PRECIZNOST")
print(met.accuracy_score(y_test, y_predicted))
mat_konf = met.confusion_matrix(y_test, y_predicted)
print(mat_konf)



plt.imshow(mat_konf)
plt.colorbar()
plt.title('Matrica konfuzije')
plt.xticks(range(3), ['3.0','4.0','5.0'])
plt.yticks(range(3), ['3.0','4.0','5.0'])
plt.show()

# klasifikacija nad treng podacim

y_predicted = clf.predict(x_train)
print("NAD TRENING PODACIMAAAAAA")
print("PRECIZNOST")
print(met.accuracy_score(y_train, y_predicted))
mat_konf = met.confusion_matrix(y_train, y_predicted)
print(mat_konf)

plt.imshow(mat_konf)
plt.colorbar()
plt.title('Matrica konfuzije')
plt.xticks(range(3), ['3.0','4.0','5.0'])
plt.yticks(range(3), ['3.0','4.0','5.0'])
plt.show()



