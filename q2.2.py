from __future__ import print_function

import nltk
nltk.download('punkt')
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF, IDF
from nltk.stem import WordNetLemmatizer

sc = SparkContext.getOrCreate()
documents = sc.textFile("/home/ansh/PycharmProjects/ICP5/data.txt").map(lambda line: line.split(" "))

lemmatizer = WordNetLemmatizer()

word_list = list(map(' '.join, documents.collect()))
word_list1 = ''
for i in word_list:
    word_list1 = word_list1 + ' ' + i
word_list2 = nltk.word_tokenize(word_list1)
lemmatized_document = ' '.join([lemmatizer.lemmatize(w) for w in word_list2])
print(lemmatized_document)

f = open("/home/ansh/PycharmProjects/ICP5/data1.txt", "w+")
f.write('' + lemmatized_document)
f.close()

document1 = sc.textFile("/home/ansh/PycharmProjects/ICP5/data.txt").map(lambda line: line.split(" "))

hashingTF = HashingTF(numFeatures=20)
tf = hashingTF.transform(document1)
tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

print("tfidf:")
for each in tfidf.collect():
    print(each)
sc.stop()