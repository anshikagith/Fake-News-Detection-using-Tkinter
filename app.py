import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import re
import string

app = Flask(__name__)

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]' %re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    news = request.form['news']
    if news:
        news = wordopt(news)
        new_x_test = [news]
        new_xv_test = v.transform(new_x_test)
        pred_LR = LR.predict(new_xv_test)
        pred_GB = GB.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_RF = RF.predict(new_xv_test)
        if pred_LR[0] == 0:
            result = "Fake News"
            color = "red"
        else:
            result = "Not Fake News"
            color = "green"
        if pred_GB[0] == 0:
            result = "Fake News"
            color = "red"
        else:
            result = "Not Fake News"
            color = "green"
        if pred_DT[0] == 0:
            result = "Fake News"
            color = "red"
        else:
            result = "Not Fake News"
            color = "green"
        if pred_RF[0] == 0:
            result = "Fake News"
            color = "red"
        else:
            result = "Not Fake News"
            color = "green"
    else:
        result = "Please enter some text"
        color = "black"
    return render_template('index.html', result=result, color=color)

if __name__ == '__main__':
    data_fake = pd.read_csv('Fake (1).csv')
    data_true = pd.read_csv('True (2).csv')
    data_fake["class"] = 0
    data_true["class"] = 1
    data_fake_manual_testing = data_fake.tail(10)
    for i in range(23480, 23470, -1):
        data_fake.drop([i], axis=0, inplace=True)
    data_true_manual_testing = data_true.tail(10)
    for i in range(21416, 21406, -1):
        data_true.drop([i], axis=0, inplace=True)
    data_fake_manual_testing_indices = data_fake.tail(10).index
    data_true_manual_testing_indices = data_true.tail(10).index

    data_fake.loc[data_fake_manual_testing_indices, 'class'] = 0
    data_true.loc[data_true_manual_testing_indices, 'class'] = 1

    data_merge = pd.concat([data_fake, data_true], axis=0)
    data = data_merge.drop(['title', 'subject', 'date'], axis=1)
    data['text'] = data['text'].apply(wordopt)
    x = data['text']
    y = data['class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    v = TfidfVectorizer()
    xv_train = v.fit_transform(x_train)
    xv_test = v.transform(x_test)
    LR = LogisticRegression()
    LR.fit(xv_train, y_train)
    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)
    GB = GradientBoostingClassifier(random_state=0)
    GB.fit(xv_train, y_train)
    RF = RandomForestClassifier(random_state=0)
    RF.fit(xv_train, y_train)
    app.run()
