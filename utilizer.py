import pandas as pd
from collections import Counter

import tensorflow as tf


# dataframe
import pandas as pd
import numpy as np


# request
import requests

# tensorflow, for NN
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.model_selection import train_test_split




# functions:
# get all of strings from sentences
def get_all_str(sentences):
    sentence = ''
    for words in sentences:
        sentence += words
    sentence = sentence.lower()
    return sentence

# get string from list
def get_str(lst):
    sentence = ''
    for char in lst:
        sentence += char+' '
    sentence = sentence.lower()
    return sentence

# get word from text
def get_word(text):
    result = nltk.RegexpTokenizer(r'\w+').tokenize(text.lower())
    return result

# remove stopwords from list
def remove_stopword(lst):
    stoplist = stopwords.words('english')
    txt = ''
    for idx in range(len(lst)):
        txt += lst[idx]
        txt += '\n'
    cleanwordlist = [word for word in txt.split() if word not in stoplist]
#     print(stoplist)
    return cleanwordlist

# lemmatize
def lemmatization(words):
    lemm = WordNetLemmatizer()
    tokens = [lemm.lemmatize(word) for word in words]
    return tokens
def preprocess(column):
    all_str = get_all_str(column)
    words = get_word(all_str)
    after_removing = remove_stopword(words)
    lemmatize = lemmatization(after_removing)
    return lemmatize

def fit_tokenizer(text, oov_token):
    tokenizer = Tokenizer(oov_token = oov_token)
    tokenizer.fit_on_texts(text)
    return tokenizer



# Function to preprocess new input text
def preprocess_input(text, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences



# Function to predict sentiment (from your previous code)
def predict_sentiment(best_model, tokenizer, text, max_len):
    print(text, '  #%^   ')
    input_data = preprocess_input(text, tokenizer, max_len)
    sentiment_proba = best_model.predict(input_data)[0][0]
    print(sentiment_proba,'##')
    if sentiment_proba >= 0.5:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'

    return sentiment

# Function to calculate the overall sentiment for a given clothing ID
def get_overall_sentiment(df,clothing_id, best_model, tokenizer, max_len):
    # Filter the reviews corresponding to the given clothing ID
    filtered_reviews = df[df['clothing_id'] == clothing_id]

    # Predict sentiment for each review
    sentiments = [predict_sentiment(best_model, tokenizer, review, max_len) for review in filtered_reviews['Text']]

    # Count the number of Positive and Negative sentiments
    sentiment_counts = Counter(sentiments)

    # Calculate overall sentiment
    if sentiment_counts['Positive'] > sentiment_counts['Negative']:
        overall_sentiment = 'Positive'
    elif sentiment_counts['Positive'] < sentiment_counts['Negative']:
        overall_sentiment = 'Negative'
    else:
        overall_sentiment = 'Neutral'  # In case of a tie

    return overall_sentiment, sentiment_counts

# Example usage:

def resulter(id):
        df = pd.read_csv('C:/Users/P.Tanmayee/Downloads/Broooooo (1)/Broooooo/Data/updated_reviews (1).csv')
        df['Text'] = df['title'] + ' ' + df['review_text']
        df['Text'] = df['Text'].apply(preprocess) 

        df.drop(['title','review_text','Division Name'],axis=1,inplace=True)
        df = df.reset_index().drop('index',axis=1)
        df['Text_Length'] = df['Text'].apply(len)

        y= df['rec_id']
        X = df.drop('rec_id',axis=1)

        # split by ratio of 0.3
        train_X, test_X, train_y, test_y = train_test_split(X['Text'], y, test_size=0.3, random_state = 8888)
        max_len = max(max(train_X.apply(len).values),max(test_X.apply(len).values))
        print(max_len,'mm')
        # Load the saved model
        new_model = tf.keras.models.load_model('C:/Users/P.Tanmayee/Downloads/Broooooo (1)/Broooooo/Models/bestmodel (1).h5')
        tokenizer = fit_tokenizer(train_X, "<OOV>")

        clothing_id = id # You can change this to any clothing ID
        overall_sentiment, sentiment_counts = get_overall_sentiment(df, clothing_id, new_model, tokenizer, max_len)
        return overall_sentiment, sentiment_counts
