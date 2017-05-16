import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from nltk import word_tokenize  
import numpy as np
from sets import Set
from sklearn import metrics
from nltk.stem import WordNetLemmatizer

csv.field_size_limit(500000)

#fungsi untuk load dataset
def load_data(dataset):
    sentences = []
    labels = []
    
    with open(dataset, 'rU') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
               text = row['text']
               type = row['type']

               sentences.append(text)
               labels.append(type)
            except:
                continue
    return sentences, labels

def loadTwitterStp(filestp):
    wordSet = Set([])
    file = open(filestp, 'r')
    for line in file:
        line = line.strip().lower()
        wordSet.add(line)
    return wordSet

stpWordSet = loadTwitterStp('twitter_stp.dic')

def defaultFilterFunc(w):
    return (w not in stpWordSet and ('http' not in w))

##how to load dataset

train_sentences, train_labels = load_data("fake_train.csv")
test_sentences, test_labels = load_data("fake_test.csv")

# for line in test_sentences:
#  print line

# count_vect = CountVectorizer(tokenizer=LemmaTokenizer())  
count_vect = CountVectorizer()  

#transform ke bentuk vector pake tf-idf
X_train_counts = count_vect.fit_transform(train_sentences)
# tfidf_transformer = TfidfTransformer(smooth_idf=False) #pake tf
# tfidf_transformer = TfidfTransformer(smooth_idf=True) #pake idf
tfidf_transformer = TfidfTransformer() #pake tfidf
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print X_train_tfidf

print 'Jumlah vocabulary di data_train:'
print count_vect.vocabulary_.get(u'algorithm')

#text classification algorithm
# clf = KNeighborsClassifier().fit(X_train_tfidf, train_labels)
clf = SGDClassifier().fit(X_train_tfidf, train_labels)

#ubah data test ke bentuk vector tfidf
X_test_counts = count_vect.transform(test_sentences)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

#prediksi data test
predicted_test = clf.predict(X_test_tfidf)
#prediksi data train
predicted_train = clf.predict(X_train_tfidf)

#print label data test
for category in predicted_test:
   print category

#cek akurasi
print 'Akurasi:'
print np.mean(predicted_test == test_labels)

# for doc, category in zip(docs_new, predicted):
#   print('%r,%s' % (doc, predicted))

## training data
#hate 226
#satire 89
#junksci 85
#state 100
#bias 418
#conspiracy 408

## testing data
#hate 20
#satire 11
#junksci 17
#state 21
#bias 22
#conspiracy 22