#!/usr/bin/env python


from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import sqlite3
import os
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import re




stopwords = nltk.corpus.stopwords.words('english')
lemmetizer=WordNetLemmatizer()


def tokenize_and_lem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    lemmas = [lemmetizer.lemmatize(t) for t in filtered_tokens]
    return lemmas



######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)

clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'model.pickle'), 'rb'))


with open(os.path.join(cur_dir,'pkl_objects','vectorizer.pickle'),'rb') as fp:
    vect = pickle.load(fp)


db = os.path.join(cur_dir, 'spambank.sqlite')

def classify(document):
    label = {0: 'ham', 1: 'spam'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()



######## Flask

app = Flask(__name__)



class smsForm(Form):
    smstext = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=5)])

@app.route('/')
def index():
    form = smsForm(request.form)
    return render_template('smsform.html', form=form)




@app.route('/results', methods=['POST'])
def results():
    form = smsForm(request.form)
    if request.method == 'POST' and form.validate():
        sms = request.form['smstext']
        y, proba = classify(sms)
        return render_template('results.html',
                                content=sms,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('smsform.html', form=form)




@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    sms = request.form['sms']
    prediction = request.form['prediction']

    inv_label = {'ham': 0, 'spam': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(sms, y)
    sqlite_entry(db, sms, y)
    return render_template('thanks.html')





if __name__ == '__main__':
    app.run(debug=True)