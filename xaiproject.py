'''Created on Apr 4, 2016
@author: Anup Kalia
'''
import pymongo
import nltk
import numpy
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
#number of rows with positive time labels (#rows = 74754)
poslen = len(pos_sentences)
#extract sentences with neg_labels
negcursor = db.xai.find({'dataPoint.label':'NEGATIVE_TIME'})
neg_sentences = [item['dataPoint']['smearedSentence'] for item in negcursor]
#number of rows with negative time labels (#rows = 5747)
neglen = len(neg_sentences)
'''
sentence = "At eight o'clock on Thursday morning"
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
'''

#---------------CONSTRUCTION OF TRAINING AND TEST DATA-------------------*/
#2/3 of negative data as training (#rows = 3850) and 1/3 of negative data as test (#rows = 1897)
training_len = int(neglen*0.67)
test_len = neglen - training_len

#construct trainingdata (#rows = 7700)
trainingdata = []
for sentence in pos_sentences[0:training_len]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence)
    trainingdata.append((tokens, "POSITIVE_TIME"))
    
for sentence in neg_sentences[0:training_len]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence)
    trainingdata.append((tokens, "NEGATIVE_TIME"))

print("TRAINING DATA CONSTRUCTED")
    
#construct testdata (#rows = 3794)
testingdata = []
for sentence in pos_sentences[training_len:neglen]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence)
    testingdata.append((tokens, "POSITIVE_TIME"))
    
for sentence in neg_sentences[training_len:neglen]:
    #tokenize each sentence
    tokens = nltk.word_tokenize(sentence)
    testingdata.append((tokens, "NEGATIVE_TIME"))

print("TEST DATA CONSTRUCTED")

#---------------EXTRACTION of FEATURES-------------------*/
def get_words_in_data(data):
    all_words = []
    for (words, labels) in data: all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_data(trainingdata))

print(word_features)

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

training_set = nltk.classify.apply_features(extract_features, trainingdata)

classifier = nltk.NaiveBayesClassifier.train(training_set)
print(classifier.show_most_informative_features(40))

testing_set = nltk.classify.apply_features(extract_features, testingdata)

print("Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

