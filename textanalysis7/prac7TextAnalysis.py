import nltk

nltk.download('punkt') #tokenizer for sentences and words
nltk.download('punkt_tab') 
nltk.download('stopwords') #predefined set of common words (e.g., "is", "the", "in")
nltk.download('wordnet') #lexical database for lemmatization
nltk.download('averaged_perceptron_tagger_eng') #POS tagger for identifying parts of speech

#initialize the text
text = "Tokenization is the first step in text analytics. The process of breaking down a text paragraph into smaller chunks such as words or sentences is called Tokenization."

#perform tokenization
from nltk.tokenize import sent_tokenize
tokenized_text = sent_tokenize(text) #sentence tokenization
print(tokenized_text)
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text) #word tokenization
print(tokenized_word)

#Removing punctuations and stop word
from nltk import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) #list of common words to ignore
print(stop_words)
text= "How to remove stop words with NLTK library in Python?"
text= re.sub('[^a-zA-Z]', ' ',text) #removes non-alphabetic characters
tokens = word_tokenize(text.lower())
filtered_text=[]
for w in tokens:
    if w not in stop_words:
        filtered_text.append(w)
print("Tokenized Sentence:",tokens)
print("Filterd Sentence:",filtered_text)

#stemming
from nltk.stem import PorterStemmer
e_words= ["wait", "waiting", "waited", "waits"]
ps=PorterStemmer()
for w in e_words:
    rootWord=ps.stem(w)
    print(rootWord)

#lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))

#applying POS tagging
from nltk.tokenize import word_tokenize
data="The pink sweater fit her perfectly"
words=word_tokenize(data)
for word in words:
    print(nltk.pos_tag([word]))

#algorithm to create representation of document by calcuating TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

documentA = 'Jupiter is the largest Planet'
documentB = 'Mars is the fourth planet from the Sun'

bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')

uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

#Initialize word count dictionaries
numOfWordsA = dict.fromkeys(uniqueWords, 0)
numOfWordsB = dict.fromkeys(uniqueWords, 0)

#Count word occurrences in Document A
for word in bagOfWordsA:
    numOfWordsA[word] += 1

#Count word occurrences in Document B
for word in bagOfWordsB:
    numOfWordsB[word] += 1

print(numOfWordsA)
print(numOfWordsB)

#calculating TF (term frequency = no of occurences of word in doc/total words in doc)

#it is a statistical measure used to evaluate how important a word is to a document in a collection or corpus

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
print(tfA)
print(tfB)

#calculating inverse document freq = total no of docs/no of docs containing the word

import math

def computeIDF(documents):
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        # Adding a small value to avoid division by zero
        idfDict[word] = math.log(N / (float(val) if val != 0 else 1e-10))
    
    return idfDict

# Recompute IDFs
idfs = computeIDF([numOfWordsA, numOfWordsB])
print(idfs)

#calculating TF-IDF = TF-IDF=TF×IDF

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

import pandas as pd

#creeating a dataframe for better visualization

df = pd.DataFrame([tfidfA, tfidfB])
df