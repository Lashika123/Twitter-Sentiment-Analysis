# Twitter Sentiment Analysis

## Overview
This project focuses on analyzing Twitter sentiment using Natural Language Processing (NLP) and Deep Learning techniques. It preprocesses tweets, converts them into numerical representations, and classifies them as **Positive, Neutral, or Negative** using a **LSTM (Long Short-Term Memory) Neural Network**.

## Features
- Cleans and preprocesses raw Twitter text data
- Uses tokenization and padding for text encoding
- Trains an LSTM model to classify tweets into sentiment categories
- Evaluates the model's performance using classification metrics

## Technologies Used
- Python
- Pandas, NumPy (Data Processing)
- NLTK (Natural Language Processing)
- TensorFlow/Keras (Deep Learning)
- scikit-learn (Data Splitting & Evaluation)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have a dataset named **Twitter_Data.csv** in the project directory.

## Explanation of the Code

### 1. Importing Necessary Libraries
```python
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report
```
- These libraries handle data processing, text cleaning, tokenization, encoding, and deep learning model creation.

### 2. Loading and Preprocessing the Data
```python
data = pd.read_csv('Twitter_Data.csv')
data.head()
```
- Loads the dataset and displays the first few rows.

```python
data['category'] = data['category'].map({-1: 'Negative', 0: 'Neutral', 1: 'Positive'})
```
- Converts sentiment labels from numeric values to human-readable labels.

```python
data.dropna(inplace=True)
```
- Removes missing values to avoid issues during model training.

### 3. Cleaning the Text Data
```python
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    return text

data['clean_text'] = data['clean_text'].apply(clean_text)
```
- Converts text to lowercase and removes special characters, leaving only words and spaces.

### 4. Tokenizing and Padding the Text
```python
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['clean_text'])

X = tokenizer.texts_to_sequences(data['clean_text'])
X = pad_sequences(X, maxlen=50)
```
- Converts text data into numerical sequences and ensures uniform input length.

### 5. Encoding Sentiment Labels
```python
encoder = LabelEncoder()
Y = encoder.fit_transform(data['category'])
```
- Converts sentiment labels into numerical format.

### 6. Splitting the Dataset
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```
- Splits the dataset into **80% training** and **20% testing** data.

### 7. Building the LSTM Model
```python
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=50))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
```
- Defines the neural network structure for sentiment classification.

### 8. Compiling and Training the Model
```python
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(X_test, Y_test))
```
- Configures and trains the model for 5 epochs.

### 9. Evaluating the Model
```python
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)

print(classification_report(Y_test, Y_pred_classes))
```
- Predicts sentiment labels for test data and prints the classification report.

### 10. Saving the Model
```python
model.save('sentiment_model.h5')
```
- Saves the trained model for future use.

## Running Real-Time Sentiment Analysis
To use the trained model for classifying new tweets:
```python
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def classify_tweet(tweet):
    model = load_model("sentiment_model.h5")
    tokenizer = Tokenizer()
    sequence = tokenizer.texts_to_sequences([tweet])
    padded = pad_sequences(sequence, maxlen=50)
    prediction = model.predict(padded)
    sentiment = np.argmax(prediction)
    labels = ['Negative', 'Neutral', 'Positive']
    print(f"Predicted Sentiment: {labels[sentiment]}")

classify_tweet("I love using this product!")
```

## License
This project is open-source and available under the MIT License.

