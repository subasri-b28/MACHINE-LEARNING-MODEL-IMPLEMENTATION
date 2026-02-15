#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

# Sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


# Sample dataset
data = {
    "text": [
        "Win money now!!!",
        "Hi, how are you?",
        "Claim your free prize",
        "Let's meet tomorrow",
        "Congratulations! You won a lottery",
        "Are we still meeting today?",
        "Free entry in 2 lakh prize draw",
        "Call me when you reach home"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

df.head()


# In[6]:


df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df.head()


# In[8]:


X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# In[9]:


model = MultinomialNB()

model.fit(X_train_tfidf, y_train)


# In[10]:


# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    "text": [
        "Win money now!!!",
        "Hi, how are you?",
        "Claim your free prize",
        "Let's meet tomorrow",
        "Congratulations! You won a lottery",
        "Are we still meeting today?",
        "Free entry in 2 lakh prize draw",
        "Call me when you reach home"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Model trained successfully ✅")


# In[ ]:




