# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:41:39 2023

@author: Administrator
"""
# Impoer Crucial Libraries

import re

# Library used for handling string operations
import string

# Data manipulation library in Python that provides data structures and tools for reading, writing, and manipulating data in tabular format
import pandas as pd

# Function in NLTK's sentiment analysis module that appends "_NEG" to words that appear between a negation word and a punctuation mark. 
from nltk.sentiment.util import mark_negation

# Corpus of commonly occurring stop words in natural language text that are removed from text during preprocessing
from nltk.corpus import stopwords

# Tokenizer in NLTK package that breaks text into words and punctuation marks
from nltk.tokenize import word_tokenize

# Stemming algorithm that removes morphological affixes from words to get their base or root form.
# Lemmatization algorithm that reduces words to their base or dictionary form.
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Feature extraction technique that converts text into a matrix of TF-IDF 
# (term frequency-inverse document frequency) features
from sklearn.feature_extraction.text import TfidfVectorizer

# Linear support vector classifier algorithm that can be used for text classification
from sklearn.svm import LinearSVC

# Function in scikit-learn that splits a dataset into training and testing data for model evaluation
from sklearn.model_selection import train_test_split

# Metric function in scikit-learn that calculates the accuracy of a model's predictions
from sklearn.metrics import accuracy_score

import pickle  

import matplotlib.pyplot as plt

import scikitplot as skplt

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, roc_auc_score

'''
The preprocess_text() function takes a Pandas Series of text data as 
input, and applies a series of text preprocessing steps to it, returning
the preprocessed text data.
'''
def preprocess_text(text):

    # Remove URLs from 'text' data
    text = text.replace(r'http\S+|https\S+|www\S+|\S+\.com\S+', '', regex=True)
    
    # Remove mentions from 'text' data
    text = text.replace(r'@\w+', '', regex=True)
    
    # Remove hashtags from 'text' data
    text = text.replace(r'#\w+', '', regex=True)

    # Remove  numeric characters from 'text' data
    text = text.replace(r'[0-9]+', '', regex=True)
    
    # Replace any sequence of repeated characters with a single instance of that character
    text = text.apply(lambda x: re.sub(r'(.)\1+', r'\1', x))
    
    # Remove punctuation from 'text' data
    text = text.str.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    text = text.apply(word_tokenize)
    
    text = text.apply(mark_negation)

    # Remove stopwords from 'text' data
    stop_words = set(stopwords.words('english'))
    text = text.apply(lambda words: [word for word in words if word not in stop_words])

    # Perform stemming and lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    text = text.apply(lambda words: [stemmer.stem(lemmatizer.lemmatize(word)) for word in words])

    return text



# Reads a CSV file named 'twitter_sentimental_analysis.csv' and stores it as a Pandas DataFrame object called dataset
dataset = pd.read_csv('twitter_sentimental_analysis.csv', encoding="ISO-8859-1", names=['target','ids','date','flag','user','text'])

# Drop rows having Nan values if exsit
dataset = dataset.dropna()

'''
Selects only the 'text' and 'target' columns from the original dataset and
creates a new DataFrame called data
'''
data = dataset.loc[:, ['text', 'target']]
'''
Replaces any target value equal to 4 (which typically corresponds 
to positive sentiment in this dataset) with 1 (to create a binary 
classification problem where 1 corresponds to positive sentiment and 
0 corresponds to negative sentiment).
'''
data['target'].replace(4, 1, inplace=True)


'''
Creates a new DataFrame called data_pos that contains only the rows 
from the original dataset where the target value is 1 
(i.e. positive sentiment).
'''
data_pos = data[data['target'] == 1]

'''
creates a new DataFrame called data_neg that contains only the rows 
from the original dataset where the target value is 0 
(i.e. negative sentiment).
'''
data_neg = data[data['target'] == 0]


# Converts all text in the 'text' column to lowercase
dataset['text']=dataset['text'].str.lower()

# Applies a pre-processing function called preprocess_text() to the 'text' column
dataset['text'] = preprocess_text(dataset['text'])

'''
Split the dataset into two separate variables independent variablr X and 
dependent variablr y.
'''
X = data.text
y = data.target

# Separating the 80% data for training data and 20% for testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

'''
The vectorizer is used to convert the text data into a matrix of 
numerical features that can be fed into a machine learning algorithm.
TF-IDF, which is a statistical measure of the importance of a word 
in a document. It also describes the ngram_range and max_features 
parameters used to specify the range of n-grams and the maximum number
of features to include in the feature matrix, respectively. 
The fit() and transform() methods are used to learn the vocabulary 
from the training data and transform both the training and test data 
into feature matrices based on that vocabulary. 
'''
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)
pickle.dump(vectoriser, open('model1.pkl','wb'))
'''
defines a Linear Support Vector Classification (LinearSVC) model and 
trains it using the training data X_train and y_train. 
The fit() method is used to fit the model to the training data. 
Once the model is trained, it is used to predict the target variable 
for the test data using the predict() method. The predicted target 
variable is stored in y_prediction.
'''
classifier = LinearSVC()
classifier.fit(X_train, y_train)
y_prediction = classifier.predict(X_test)

# Computes the accuracy of the trained machine learning model
acc = accuracy_score(y_test, y_prediction)
print('Accuracy:', acc*100,'%')

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Confusion Matrix for Classification
# skplt.metrics.plot_confusion_matrix(y_test, y_prediction, normalize=False)

# Classification Report for Voting Classifier
print(f'Classification Report:\n{classification_report(y_test, y_prediction)}')

# Predict class probabilities for the test data
y_prob = classifier.decision_function(X_test)

# Calculate the false positive rate and true positive rate
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Calculate the AUC score
auc_score = roc_auc_score(y_test, y_prob)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc_score))
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for LinearSVC')
plt.legend(loc='lower right')
plt.show()