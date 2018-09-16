import pandas as pd
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

df = pd.read_csv("../../datasets/names_dataset.csv")
df['first_alpha'] = '0'
df['last_alpha'] = '0'
df['len'] = 0
df.set_index('index', inplace = True)
def processNames(row):
    row['first_alpha'] = row['name'][0]
    row['last_alpha'] = row['name'][-1]
    row['len'] = len(row['name'])
    return row

df = df.apply(processNames, axis=1)

features = df[['first_alpha', 'last_alpha', 'len']].values
goal = df['sex'].values
encoder = LabelEncoder()
alpha = ['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
         'a', 'b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
encoder.fit(alpha)
features[:, 0] = encoder.transform(features[:, 0])
features[:, 1] = encoder.transform(features[:, 1])

x_train, x_test, y_train, y_test = train_test_split(features, goal, test_size=0.2, random_state=0)

clf = LinearSVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(clf.score(x_test, y_test))
print(classification_report(y_test, y_pred))


RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
PICKLE_DIR = os.path.join(RES_DIR, 'model.pkl')
ENCODER_DIR = os.path.join(RES_DIR, 'encoder.pkl')

joblib.dump(clf, PICKLE_DIR)
joblib.dump(encoder, ENCODER_DIR)
