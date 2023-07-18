# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 17:00:29 2023

@author: aitza
"""
import re
import string
import pandas as pd
from nltk.sentiment.util import mark_negation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle

# Load the trained sentiment analysis model
model = pickle.load(open('classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def preprocess_text(text):
    # Remove URLs from 'text' data
    text = text.replace(r'http\S+|https\S+|www\S+|\S+\.com\S+', '', regex=True)
    
    # Remove mentions from 'text' data
    text = text.replace(r'@\w+', '', regex=True)
    
    # Remove hashtags from 'text' data
    text = text.replace(r'#\w+', '', regex=True)

    # Remove numeric characters from 'text' data
    text = text.replace(r'[0-9]+', '', regex=True)
    
    # Replace any sequence of repeated characters with a single instance of that character
    text = text.apply(lambda x: re.sub(r'(.)\1+', r'\1', x))
    
    # Remove punctuation from 'text' data
    text = text.str.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    text = text.apply(word_tokenize)
    
    # Apply negation marking to the text
    text = text.apply(mark_negation)

    # Remove stopwords from 'text' data
    stop_words = set(stopwords.words('english'))
    text = text.apply(lambda words: [word for word in words if word not in stop_words])

    # Perform stemming and lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    text = text.apply(lambda words: [stemmer.stem(lemmatizer.lemmatize(word)) for word in words])

    return text

# Sample data
data = {
    "tweet": ["Oscar rubbing my feet w/ alcohol at this time to help with the pain"]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)
df['tweet'] = df['tweet'].map(str.lower)

# Preprocess the text
preprocessed_text = preprocess_text(df['tweet'])

# Convert preprocessed text to string
preprocessed_text_str = preprocessed_text.astype(str)

# Vectorize the preprocessed text
vectorized_text = vectorizer.transform(preprocessed_text_str)

# Make prediction using the loaded model
prediction = model.predict(vectorized_text)

# Print the sentiment
if prediction == 0:
    print("Positive sentiment tweet:", df['tweet'][0])
else:
    print("Negative sentiment tweet:", df['tweet'][0])



