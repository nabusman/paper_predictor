import sys
import os
import subprocess
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.mllib.recommendation import MatrixFactorizationModel
sys.path.append("/work")
from paper_predictor import clean_text

def recommend_papers(rawInput, numPapers = 10):
  # Spark setup
  sc = SparkContext(master = "yarn", appName = "PaperRecommendation")
  sc.setLogLevel("FATAL")
  sqlContext = SQLContext(sc)
  print "Cleaning input text"
  inputDoc = clean_text(rawInput)
  contentRDD = sqlContext.read.load("/content").rdd.map(lambda x: ' '.join(x[0]))
  hashingTF = HashingTF()
  print "Creating TF-IDF with new input"
  tf = hashingTF.transform(contentRDD)
  idf = IDF(minDocFreq=2).fit(tf)
  tfidfInputDoc = idf.transform(hashingTF.transform(' '.join(inputDoc)))
  # Take the top 5 terms and submit them to ALS for user (document) recommendations
  def top_n_terms(sparseVec, n):
    termFreqNP = np.column_stack((sparseVec.indices, sparseVec.values))
    tfSorted = np.sort(termFreqNP.view(dtype = [('term', 'f8'), ('score', 'f8')]), order = 'score', axis = 0)[::-1]
    topTerms = tfSorted[0:n]["term"].tolist() if len(tfSorted) > n else tfSorted[0:len(tfSorted) - 1]["term"].tolist()
    return map(lambda x: int(x[0]), topTerms)
  print "Extracting top " + str(numPapers) + " terms"
  topTerms = top_n_terms(tfidfInputDoc, numPapers)
  topTerms = filter(lambda x: tfidfInputDoc[x] != 0, topTerms)
  if topTerms == []:
    return []
  # Return list of numPapers documents
  # Load ALS model from file
  print "Loading ALS model from HDFS and predicting papers"
  model = MatrixFactorizationModel.load(sc, "/ALS")
  recommendations = []
  for x in range(numPapers):
    recommendations.extend(model.recommendUsers(topTerms[x % len(topTerms)], 1))
  recommendations = map(lambda x: x.user, recommendations)
  # Load idDF from file
  print "Loading document ID's and creating URLS of recommended papers"
  idRDD = sqlContext.read.load("/docID").rdd.zipWithIndex().map(lambda x: Row(**{"docID" : x[1], 
    "docName" : x[0].asDict()["docName"], "docURL" : x[0].asDict()["docURL"]}))
  idDF = sqlContext.createDataFrame(idRDD)
  # Extract the URLs from idDF
  papers = map(lambda x: x["docURL"], idDF.filter(idDF["docID"].isin(recommendations)).select("docURL").collect())
  print ' '.join(papers)

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print "File path to raw input needed!"
  else:
    rawInputFilePath = sys.argv[1]
    with open(rawInputFilePath) as inputFile:
      rawInput = inputFile.readlines()
    recommend_papers(rawInput[0])
    os.remove(rawInputFilePath)