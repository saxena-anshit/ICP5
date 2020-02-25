from __future__ import print_function
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF, IDF

sc = SparkContext.getOrCreate()
documents = sc.textFile("/home/ansh/PycharmProjects/ICP5/data.txt").map(lambda line: line.split(" "))

hashingTF = HashingTF(numFeatures=20)
tf = hashingTF.transform(documents)

tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

print("tfidf:")
for each in tfidf.collect():
    print(each)
sc.stop()
