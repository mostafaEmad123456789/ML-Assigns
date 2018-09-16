import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

positive_files = os.listdir("../../datasets/Twitter/Twitter/Positive")
negative_files = os.listdir("../../datasets/Twitter/Twitter/Negative")

lines = []
for p_file, n_file in zip(positive_files, negative_files):
    try:
        pf = open("../../datasets/Twitter/Twitter/Positive/"+p_file, 'r', encoding='utf-8')
        nf = open("../../datasets/Twitter/Twitter/Negative/"+n_file, 'r', encoding='utf-8')
        text = pf.read()
        lines.append([text.strip(), "Positive"])
        text = nf.read()
        lines.append([text.strip(), "Negative"])
        pf.close()
        nf.close()
    except Exception as e:
        pass
lines = np.array(lines)

df = pd.DataFrame({"text":lines[:, 0], "label":lines[:, 1]})

x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'] , test_size=0.2, random_state=0)
print(len(y_train))
print(len(y_test))

vectorizer = CountVectorizer()

vectorizer.fit(x_train)

x_train_bow = vectorizer.transform(x_train)
x_test_bow = vectorizer.transform(x_test)

clf = SVC(kernel='linear', probability=True)
clf.fit(x_train_bow, y_train)
y_pred = clf.predict(x_test_bow)
print(classification_report(y_test, y_pred))

accuracies = cross_val_score(estimator = clf, X = x_train_bow, y = y_train, cv = 10)
print("mean of acc:", accuracies.mean())
print("Standard diviation of acc:", accuracies.std())
print("model acc : {:.2f} (+/- {:.2f})%".format(accuracies.mean(), accuracies.std()))

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
PICKLE_DIR = os.path.join(RES_DIR, 'model.pkl')
VECTORIZER_DIR = os.path.join(RES_DIR, 'vectorizer.pkl')

joblib.dump(clf, PICKLE_DIR)
joblib.dump(vectorizer, VECTORIZER_DIR)
