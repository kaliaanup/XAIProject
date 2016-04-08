'''Created on Apr 4, 2016
@author: Anup Kalia
'''

import pymongo
import nltk
import itertools
import random
from utilities import (get_unigrams, get_bigrams, remove_stopwords_unigrams, remove_stopwords_bigrams, 
                      lemmatize_tokens, extract_sentiments, contains_positive_word, contains_negative_word,
                      contains_multiple_days, contains_multiple_months, count_multiple_time_of_days, get_classifier,
                      get_accuracy_measures)
#from nltk.corpus import treebank

#------------CONNECT TO MONGODB-------------------------*/
client = pymongo.MongoClient("localhost", 27017)
#db name is dbxai
db = client.dbxai
print("CONNECTED TO MONGODB "+db.name)

#-------------ACCESS EACH ROW from MONGODB---------------------------*/
#collection name is xai
#extract sentences with pos_labels
poscursor = db.xai.find({'dataPoint.label':'POSITIVE_TIME'})
pos_sentences = [item['dataPoint']['smearedSentence'] for item in poscursor]
#for random subsampling
random.shuffle(pos_sentences)

#number of rows with positive time labels (#rows = 74754)
poslen = len(pos_sentences)
#extract sentences with neg_labels
negcursor = db.xai.find({'dataPoint.label':'NEGATIVE_TIME'})
neg_sentences = [item['dataPoint']['smearedSentence'] for item in negcursor]
#for random subsampling
random.shuffle(neg_sentences)
#number of rows with negative time labels (#rows = 5747)
neglen = len(neg_sentences)

#---------------CONSTRUCTION OF TRAINING AND TEST DATA-------------------*/
#2/3 of negative data as training (#rows = 3850) and 1/3 of negative data as test (#rows = 1897)
training_len = int(neglen*0.67)
test_len = neglen - training_len

#construct trainingdata (#rows = 7700)
trainingdata = []
for sentence in pos_sentences[0:training_len]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence.lower())
    tokens = lemmatize_tokens(tokens)
    trainingdata.append((tokens, "POSITIVE_TIME"))
    
for sentence in neg_sentences[0:training_len]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence.lower())
    tokens = lemmatize_tokens(tokens)
    trainingdata.append((tokens, "NEGATIVE_TIME"))

'''
#oversampling the rare class
for sentence in neg_sentences[0:training_len]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence.lower())
    tokens = lemmatize_tokens(tokens)
    trainingdata.append((tokens, "NEGATIVE_TIME"))
    
for sentence in neg_sentences[0:training_len]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence.lower())
    tokens = lemmatize_tokens(tokens)
    trainingdata.append((tokens, "NEGATIVE_TIME"))
'''
print("TRAINING DATA CONSTRUCTED")
print(len(trainingdata))
    
#construct testdata (#rows = 3794)
testingdata = []
for sentence in pos_sentences[training_len:neglen]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence)
    tokens = lemmatize_tokens(tokens)
    testingdata.append((tokens, "POSITIVE_TIME"))
    
for sentence in neg_sentences[training_len:neglen]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence)
    tokens = lemmatize_tokens(tokens)
    testingdata.append((tokens, "NEGATIVE_TIME"))

'''
#oversampling the rare class
for sentence in neg_sentences[training_len:neglen]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence)
    tokens = lemmatize_tokens(tokens)
    testingdata.append((tokens, "NEGATIVE_TIME"))

for sentence in neg_sentences[training_len:neglen]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence)
    tokens = lemmatize_tokens(tokens)
    testingdata.append((tokens, "NEGATIVE_TIME"))
'''
print(len(testingdata))
print("TEST DATA CONSTRUCTED")

#---------------EXTRACTION of FEATURES-------------------*/

#get all unigrams
unigrams = get_unigrams(trainingdata)
#get all unigrams with no stop words
#unigrams_no_stopwords = remove_stopwords_unigrams(trainingdata)
#get all bigrams 
bigrams = get_bigrams(trainingdata)

#document takes the tokens
def extract_features(document):
    data_words = set(document)
    features = {}
    #add unigrams or unigrams_no_stopwords
    for w in unigrams:
        #check if the word exist in a sentence and then mark true or false
        features[w] = (w in data_words) 
    #add bigrams or bigrams_no_stopwords
    for ngram in bigrams:
       features[ngram] = (ngram in itertools.chain(data_words)) 
    #check avg length of sentences--positive is 8.96 and negative is 14.17
    ''' 
    if(len(data_words) >= 14):
        features["longlen"] = "True"
        features["shortlen"] = "False"
    elif(len(data_words) <= 9):
        features["shortlen"] = "True"
        features["longlen"] = "False"
    else:
        features["shortlen"] = "False"
        features["longlen"] = "False"
    '''
    #check sentiments of sentences (negative) or (positive)
    sentiment = extract_sentiments(data_words)
    if(sentiment == 'pos'):
        features["positiveSent"] = "True"
        features["negativeSent"] = "False"
        features["neutralSent"] = "False"
    elif(sentiment == 'neg'):
        features["positiveSent"] = "False"
        features["negativeSent"] = "True"
        features["neutralSent"] = "False"
    else:
        features["positiveSent"] = "False"
        features["negativeSent"] = "False"
        features["neutralSent"] = "True"
    #contains positive sentiment words
    features["containPositiveWord"] = contains_positive_word(data_words)
    #contains negative word
    #features["containNegativeWord"] = contains_negative_word(data_words)
    #contains multiple months
    '''
    #features["containMultipleDays"]  = contains_multiple_days(data_words)
    #contains multiple days
    features["containMonthDays"] = contains_multiple_months(data_words)
    '''
    #contains multiple time
    features["containTimes"] = count_multiple_time_of_days(data_words)
    return features  
#----------------------------CLASSIFICATION----------------------------------*/
training_set = nltk.classify.apply_features(extract_features, trainingdata)
testing_set = nltk.classify.apply_features(extract_features, testingdata)


#----------------------------Naive Bayes----------------------------------*/
#nbClassifier = nltk.NaiveBayesClassifier.train(training_set)

classifier = get_classifier(training_set,"Naive_Bayes")
presult = get_accuracy_measures(classifier, testing_set, "POSITIVE_TIME")
print(presult)
nresult = get_accuracy_measures(classifier, testing_set, "NEGATIVE_TIME")
print(nresult)

''''
classifier = get_classifier(training_set,"Maximum_Entropy")
presult = get_accuracy_measures(classifier, testing_set, "POSITIVE_TIME")
print(presult)
nresult = get_accuracy_measures(classifier, testing_set, "NEGATIVE_TIME")
print(nresult)
'''
'''
classifier = get_classifier(training_set,"Decision_Tree")
presult = get_accuracy_measures(classifier, testing_set, "POSITIVE_TIME")
print(presult)
nresult = get_accuracy_measures(classifier, testing_set, "NEGATIVE_TIME")
print(nresult)
'''
#print(classifier.show_most_informative_features(40))

#-----------------------------COMPUTING PERFORMANCE of CLASSIFIERS-------------------------------*/

