import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn import metrics
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

csv.field_size_limit(500000)

#fungsi untuk load dataset
def load_data(dataset):
    sentences = []
    labels = []
    wordnet_lemmatizer = WordNetLemmatizer()
    thesentence=""
    with open(dataset, 'rU') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
               # text = row['text']
               words_tokenize = word_tokenize(row['text'])
               # print words_tokenize
               for words in words_tokenize:
                text = wordnet_lemmatizer.lemmatize(words,'v')
                thesentence = thesentence+" "+text            
                # print text+"\n"
                # text = wordnet_lemmatizer.lemmatize(row['text'], 'v')
               # print text+"\n"
               thesentence+="\n"
               # print thesentence
               type = row['type']
               sentences.append(thesentence)
               labels.append(type)
            except:
                continue
    return sentences, labels


##how to load dataset

train_sentences, train_labels = load_data("fake_train.csv")
test_sentences, test_labels = load_data("fake_test.csv")
# print train_sentences
#Tokenizing text
count_vect = CountVectorizer()

#transform ke bentuk vector pake tf-idf
X_train_counts = count_vect.fit_transform(train_sentences)
# tfidf_transformer = TfidfTransformer(smooth_idf=False) #pake tf
# tfidf_transformer = TfidfTransformer(smooth_idf=True) #pake idf
tfidf_transformer = TfidfTransformer() #pake tfidf
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print X_train_tfidf
print X_train_counts.shape
print "Algoritma yang digunakan adalah SGDClassifier"
print "Menggunakan TFIDF"
print 'Jumlah vocabulary di data_train:'
print count_vect.vocabulary_.get(u'algorithm')

#text classification algorithm
clf = SGDClassifier().fit(X_train_tfidf, train_labels)

#ubah data test ke bentuk vector tfidf
X_new_counts = count_vect.transform(test_sentences)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#prediksi data test
predicted = clf.predict(X_new_tfidf)

#print label data test
for category in predicted:
   print category

#cek akurasi
X_old_counts = count_vect.transform(train_sentences)
X_old_tfidf = tfidf_transformer.transform(X_old_counts)
predicted_train = clf.predict(X_old_tfidf)
print 'Akurasi:'
print np.mean(predicted == test_labels)

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
