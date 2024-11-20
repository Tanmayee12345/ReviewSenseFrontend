from flask import Flask, request, render_template, jsonify
import pandas as pd
from keras.models import load_model
import pickle
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
import tensorflow as tf
# Load the saved model
model = tf.keras.models.load_model('Models/bestmodel.h5')
df = pd.read_csv('Data/updated_reviews (1).csv')

# Preprocessing functions
def get_all_str(sentences):
    """Combines a list of sentences into one lowercase string."""
    sentence = ''
    for words in sentences:
        sentence += words
    return sentence.lower()

def get_str(lst):
    sentence = ''
    for char in lst:
        sentence += char+' '
    sentence = sentence.lower()
    return sentence

def get_word(text):
    """Tokenizes the input text into words."""
    result = nltk.RegexpTokenizer(r'\w+').tokenize(text.lower())
    return result

def remove_stopword(lst):
    stoplist = stopwords.words('english')
    txt = ''
    for idx in range(len(lst)):
        txt += lst[idx]
        txt += '\n'
    cleanwordlist = [word for word in txt.split() if word not in stoplist]
#     print(stoplist)
    return cleanwordlist

def lemmatization(words):
    lemm = WordNetLemmatizer()
    tokens = [lemm.lemmatize(word) for word in words]
    return tokens

def Freq_df(cleanwordlist):
    Freq_dist_nltk = nltk.FreqDist(cleanwordlist)
    df_freq = pd.DataFrame.from_dict(Freq_dist_nltk, orient='index')
    df_freq.columns = ['Frequency']
    df_freq.index.name = 'Term'
    df_freq = df_freq.sort_values(by=['Frequency'],ascending=False)
    df_freq = df_freq.reset_index()
    return df_freq

def preprocess(column):
    """Preprocess text by combining, tokenizing, removing stopwords, and lemmatizing."""
    all_str = get_all_str(column)
    words = get_word(all_str)
    after_removing = remove_stopword(words)
    lemmatize = lemmatization(after_removing)
    frequency_df = Freq_df(lemmatize)
    return frequency_df

# Load dataset for initial analysis (optional)
df = pd.read_csv("Data/updated_reviews (1).csv")
df.dropna(subset=['review_text', 'Division Name', 'title'], inplace=True)
df['Text'] = df['title'] + ' ' + df['review_text']
df.drop(['title', 'review_text', 'Division Name'], axis=1, inplace=True)
df = df.reset_index().drop('index', axis=1)
df['Text_Length'] = df['Text'].apply(len)

# Preprocess the 'Text' column for the dataset (optional)
df['Text'] = df['Text'].apply(preprocess)

y= df['rec_id']
X = df.drop('rec_id',axis=1)
# split by ratio of 0.3
train_X, test_X, train_y, test_y = train_test_split(X['Text'], y, test_size=0.3, random_state = 8888)
max_len = max(max(train_X.apply(len).values),max(test_X.apply(len).values))
# for tokenization
def fit_tokenizer(text, oov_token):
    tokenizer = Tokenizer(oov_token = oov_token)
    tokenizer.fit_on_texts(text)
    return tokenizer

# for sequence, padding
def seq_padding(sentences, tokenizer, padding, truncating, maxlen):
    sequences = tokenizer.texts_to_sequences(sentences)
    pad_trunc_sequences = pad_sequences(sequences, padding = padding, maxlen = maxlen, truncating=padding)
    return pad_trunc_sequences
# for tokenization
tokenizer = fit_tokenizer(train_X, "<OOV>")

word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)

train_X = seq_padding(train_X,tokenizer, 'post', 'post',max_len)
test_X = seq_padding(test_X,tokenizer, 'post', 'post', max_len)
GLOVE_FILE = '/content/drive/MyDrive/glove.6B.100d.txt'
GLOVE_EMBEDDINGS = {}

with open(GLOVE_FILE) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        GLOVE_EMBEDDINGS[word] = coefs

EMBEDDINGS_MATRIX = np.zeros((VOCAB_SIZE+1, 100))

num = 0
missed_list = []

for word, i in word_index.items():
    embedding_vector = GLOVE_EMBEDDINGS.get(word)
    if embedding_vector is not None:
        EMBEDDINGS_MATRIX[i] = embedding_vector
    else:
        num += 1
        missed_list.append(word)

print('How many missed words? ',num)
print('As example: ',missed_list[:10])

print(EMBEDDINGS_MATRIX.shape)
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Import CountVectorizer
import pandas as pd

# Function to create Bag of Words representation
def create_bow_representation(corpus):
    # Join list elements into a single string if they are lists
    corpus = [' '.join(doc) if isinstance(doc, list) else doc for doc in corpus]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df

# Assuming df is your DataFrame
bow_representation = create_bow_representation(df['Text'])
VOCAB_SIZE = len(word_index)
print(VOCAB_SIZE)
EMBEDDING_DIM = len(next(iter(GLOVE_EMBEDDINGS.values())))
print(EMBEDDING_DIM)

tokenizer = fit_tokenizer(train_X, "<OOV>")

# Function to preprocess new input text
def preprocess_input(text, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences


def predict_sentiment(best_model, tokenizer, text, max_len):
    input_data = preprocess_input(text, tokenizer, max_len)
    sentiment_proba = best_model.predict(input_data)[0][0]

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

    # return overall_sentiment, sentiment_counts
    return overall_sentiment, dict(sentiment_counts)

# Flask routes
@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        data = request.get_json()
        clothing_id = data.get('clothing_id')
        
        if not clothing_id:
            return jsonify({"error": "No clothing_id provided"}), 400

        # Get overall sentiment for the given clothing ID
        overall_sentiment, sentiment_counts = get_overall_sentiment(df,clothing_id,model,tokenizer,71)

        print("overall_sentiment ",overall_sentiment)
        # Return JSON response with sentiment and counts
        return jsonify({
            "clothing_id": clothing_id,
            "predicted_sentiment": overall_sentiment,
            "sentiment_counts": sentiment_counts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
