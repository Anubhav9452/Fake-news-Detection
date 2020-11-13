import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# now let's read the data int dataframe and get the shape of data in 5 records
#Read the data
df=pd.read_csv('news.csv')

#Get shape and head
df.shape
df.head()

# get labels from dataframe
labels=df.label
labels.head()

# splitting the data into train test split
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# fit and transform the vectorizer on the train set and transform the vectorizer on the test set

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

# prediction of test set with tfidfvectorizer and calculating accuracy
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#  print  confusion matrix to gain insight into number of false and true positives and negatives
print(confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']))

