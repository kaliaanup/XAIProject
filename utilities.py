'''
Created on Apr 4, 2016

@author: Anup Kalia
'''
import nltk
from nltk.util import bigrams
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string


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

wnl = WordNetLemmatizer()

#-----------------Lemmatize the tokens------------------------------*/
def lemmatize_tokens(tokens):
    lemma_tokens = []
    for word in tokens:
        lemma_tokens.append(wnl.lemmatize(word))

    return lemma_tokens

#-----------------Extract Sentiment of sentences-------------------*/
sid = SentimentIntensityAnalyzer()
def extract_sentiments(tokens):
    sentence="".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
    ss = sid.polarity_scores(sentence)
    max =0; 
    sentiment=""
    for k in sorted(ss):
        if(k=='neg'):
            neg=ss[k]
        if(k=='pos'):
            pos=ss[k]
        if(k=='neu'):
            neu=ss[k]
            
    if(neu > pos and neu> neg):
        if(pos == neg):
            sentiment = 'neu'
        else:
            if(pos>neg):
                sentiment = 'pos'
            else:
                sentiment = 'neg'
                
    return sentiment

#-----------------Check for Positive Words-------------------*/
def contains_positive_word(tokens):
    pos_word = ["great", "work", "fine", 'good', 'okay']
    for word in tokens:
            if word in pos_word:
                return "True"
            else:
                return "False"

#-----------------Check for Negative Words-------------------*/
def contains_negative_word(tokens):
    for word in tokens:
            if "not" in word  or "n't" in word:
                return "True"
            else:
                return "False"

#---------------Contains multiple days-----------------------*/
def contains_multiple_days(tokens):
    days = ["monday", "mon", "tuesday", "tue", "wednesday", "wed", "thursday", 
            "thu", "friday", "fri", "saturday", "sat", "sunday", "sun"]
    count=0
    for word in tokens:
        if word in days:
            count = count + 1
    if count>1:
        return "True"
    else:
        return "False"
#---------------Contains multiple months-----------------------*/
def contains_multiple_months(tokens):
    months = ["january", "jan", "february", "feb", "march", "mar", "april", "apr", 
            "may", "june", "jun", "july", "jul", "august", "aug", "september", 
            "sept", "october", "oct", "november", "nov", "december", "dec"]
    count=0
    for word in tokens:
        if word in months:
            count = count + 1
    if count>1:
        return "True"
    else:
        return "False"

#---------------Contains multiple time frame-----------------------*/
def count_multiple_time_of_days(tokens):
    time = ["today", "afternoon", "tomorrow", "evening", "morning", "tonight"]
    count=0
    for word in tokens:
        if word in time:
            count = count + 1
    if count>1:
        return "True"
    else:
        return "False"
