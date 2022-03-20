import pandas as pd                          # Data Processing Libraries
import numpy as np
import matplotlib.pyplot as plt

import newspaper
from newspaper import Article

import nltk  #(Natural Language Tool Kit - NLTK)                               # NLP Libraries
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator # Libaray for Word Visualization

import os
import time

filepath = "static/cloud.png"

if os.path.exists(filepath):
    os.remove(filepath)
else:
    pass


# Preprocessing functions of Text Data

stop_words = set(stopwords.words('english')) # Stopwords function
wn = WordNetLemmatizer()        # lemmatization

wc = WordCloud() #background_color="white"


from flask import Flask, request, jsonify, render_template
from flask import Markup
import joblib
import webbrowser

model = joblib.load('finalmodel_logreg.pkl') # Trained Model File
vector = joblib.load('tfidf_vectorizer.pkl') # Trained Word Embedding file

# Functions

def formaturl(url):
    if not re.match('(?:http|ftp|https)://', url):
        return 'http://{}'.format(url)
    return url

app = Flask(__name__) #Initialize the flask App

@app.route('/')
def home():
    return render_template("frontpage.html")


@app.route('/predict',methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        link = request.form['newsurl']
    except:
        link = np.nan

    print(link)

    if link is not np.nan:
        url = formaturl(link)
        article = Article(url)
        article.download()
        article.parse()
        if article.text != "":
            print(url,"success")
            article = article.text
    else:
        article = request.form['newstext']

    print(article)

    # split into words
    tokens = word_tokenize(article)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [w for w in words if not w in stop_words]
    # Lemmatization of words (Root Word Retrieval)
    lemwords = [wn.lemmatize(word) for word in words]
    # Removing Duplicated Words
    impwords = set(lemwords)
    # Joining the words
    impwords = ' '.join(impwords)

    ## Prediction

    article_tfidf = vector.transform([impwords]).toarray()
    probs = model.predict_proba(article_tfidf)
    predclass = np.where(model.predict(article_tfidf) == 0, 'FAKE', 'REAL')
    predprob = (probs[0].max())*100

    # Generate a word cloud image
    stopwords = set(STOPWORDS)
    mask = np.array(Image.open("static/mask.png"))
    wordcloud = WordCloud(stopwords=stopwords,background_color='white', max_words=1000, mask=mask,contour_color='#023075',contour_width=3,colormap='rainbow').generate("".join(impwords))
    # create image as cloud
    plt.imshow(wordcloud, interpolation="bilinear")
    # store to file
    plt.savefig("static/cloud.png", format="png")
    
    return render_template('frontpage.html', urlgiven = link, articletext = article, pclass = predclass, pprob = round(predprob,2))

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000, debug = True)