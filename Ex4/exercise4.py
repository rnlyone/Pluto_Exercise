import pandas as pd
from sklearn.svm import SVC

import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import accuracy_score

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.model_selection import cross_val_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# prompt: make me the slangremover function for english language with working slang_dict from github

slang_dict = {
    "afaik": "as far as I know",
    "brb": "be right back",
    "btw": "by the way",
    "lol": "laugh out loud",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "imo": "in my opinion",
    "idk": "I don't know",
    "rofl": "rolling on the floor laughing",
    "wtf": "what the f***",
    "lmfao": "laughing my f***ing ass off",
    "asap": "as soon as possible",
    "tbh": "to be honest",
    "diy": "do it yourself",
    "np": "no problem",
    "thx": "thanks",
    "pls": "please",
    "yolo": "you only live once",
    "gtg": "got to go",
    "btw": "by the way",
    "irl": "in real life",
    "omg": "oh my god",
    "lmk": "let me know",
    "imo": "in my opinion",
    "afaik": "as far as I know",
    "rofl": "rolling on the floor laughing",
    "lmao": "laughing my ass off",
    "tbh": "to be honest",
    "brb": "be right back",
    "idk": "I don't know",
    "ily": "I love you",
    "jk": "just kidding",
    "thx": "thanks",
    "asap": "as soon as possible",
    "nvm": "nevermind",
    "np": "no problem",
    "pls": "please",
    "ttyl": "talk to you later",
    "lol": "laughing out loud",
    "omg": "oh my god",
    "wtf": "what the f***",
    "gtg": "got to go",
    "btw": "by the way",
    "irl": "in real life",
    "lmk": "let me know",
    "imo": "in my opinion",
    "afaik": "as far as I know",
    "rofl": "rolling on the floor laughing",
    "lmao": "laughing my ass off",
    "tbh": "to be honest",
    "brb": "be right back",
    "idk": "I don't know",
    "ily": "I love you",
    "jk": "just kidding",
    "thx": "thanks",
    "asap": "as soon as possible",
    "nvm": "nevermind",
    "np": "no problem",
    "pls": "please",
    "ttyl": "talk to you later"
}

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers

    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text
    return text

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower()
    return text

import spacy


nlp = spacy.blank("en") 

def tokenizingText(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]  
    return tokens

def deSlangText(text):
    new_text = []
    for word in text.split():
        if word.lower() in slang_dict:
            new_text.append(slang_dict[word.lower()])
        else:
            new_text.append(word)
    return " ".join(new_text)

def filteringText(text):  # Remove stopwords in a text
    listStopwords = set(stopwords.words('english'))
    filtered = [w for w in text if not w.lower() in listStopwords]
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text

# Lemmatizer object
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
    return lemmatized_words

def stemmingText(text):  # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence


if __name__ == "__main__":
    df = pd.read_csv('output_sentences.csv')
    df['text_clean'] = df['sentence'].astype(str).apply(cleaningText)
    df['casefolding'] = df['text_clean'].apply(casefoldingText)
    df['text_deslanged'] = df['casefolding'].apply(deSlangText)
    df['text_preprocessed'] = df['text_deslanged'].apply(tokenizingText)
    df['text_filtered'] = df['text_preprocessed'].apply(filteringText)
    df['text_stemmed'] = df['text_filtered'].apply(lemmatize_text)
    df['text_classifier'] = df['text_filtered'].apply(toSentence)

    tf_idf_ngram_vectorizer = TfidfVectorizer()
    X = df['text_classifier']
    y = df['sentiment']

    X_bi = tf_idf_ngram_vectorizer.fit_transform(X)

    # Convert sparse matrix to dense and then to PyTorch tensors
    X_bi_dense = X_bi.toarray()
    X_tensor = torch.tensor(X_bi_dense, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.4, random_state=42)

    # Create DataLoader for training and test sets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    input_dim = X_bi_dense.shape[1]
    hidden_dim = 128
    output_dim = 1

    model = LSTMModel(input_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted = (outputs.squeeze() > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
