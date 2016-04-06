'''
Created on Apr 4, 2016

@author: Anup Kalia
'''
import nltk
from nltk.util import bigrams
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

#-----------store all words as unigrams----------------------*/
def get_unigrams(data):
    all_words = []
    for (words, label) in data:
        for word in words:
            all_words.append(word)
            
    wordlist = nltk.FreqDist(all_words)
    unigrams =  wordlist.keys()
    
    return unigrams

#-----------remove stopwords from unigrams-------------------*/
def remove_stopwords_unigrams(data):
    all_words = []
    stopset = set(stopwords.words('english'))
    
    for (words, label) in data:
        for word in words:
            if word not in stopset:
                all_words.append(word)
            
    wordlist = nltk.FreqDist(all_words)
    unigrams =  wordlist.keys()
    return unigrams

#----------store all bigrams--------------------------------*/
def get_bigrams(data):
    all_words = []
    for (words, label) in data:
        for word in words: 
            all_words.append(word)
            
    bigram_finder = BigramCollocationFinder.from_words(all_words)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 1000)   
    wordlist = nltk.FreqDist(bigrams)
    bigrams = wordlist.keys()
    
    return bigrams

#-----------remove stopwords from bigrams-------------------*/    
def remove_stopwords_bigrams(data):
    all_words = []
    stopset = set(stopwords.words('english'))
    
    for (words, label) in data:
        for word in words:
            if word.lower() not in stopset:
                all_words.append(word)
                
    bigram_finder = BigramCollocationFinder.from_words(all_words)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 1000)   
    wordlist = nltk.FreqDist(bigrams)
    bigrams = wordlist.keys()
    
    return bigrams

