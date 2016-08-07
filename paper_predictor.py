import re
import nltk
import string
import subprocess
import numpy as np
from pyspark.sql import Row
from nltk.corpus import stopwords
from nltk.stem.porter import *
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# TO DO:
# - Save /content as a space seperated string rather than as a list
# - Look into saving and loading the IDF from file, or explore alternative TF-IDF implementations that can be pickeled
# - Research into parallelizing NLTK functions so that clean_text can be mapped and hence parallelized
# - Test accuracy of recommendations
# - Improve accuracy by changing ALS parameters

# REGEX Compile
pathRegex = re.compile(r'(s3.*)')
fileNameRegex = re.compile(r'\/(arXiv_src.*)')
folderNameRegex = re.compile(r'src\_(.*?)\_')
arxivIDRegex = re.compile(r'\/work\/.*?\/(.*?)\/')
docURLRegex = re.compile(r'([A-Za-z-]+)(\d+)')

# NLTK_Data load
nltk.data.path.append('/work/nltk_data')

def load_s3_filelist():
  """Extract s3 file list from arXiv"""
  cmd = ['/usr/local/bin/s3cmd', 'ls', 's3://arxiv/src/', '--requester-pays']
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  out, err = p.communicate()
  file_list = []
  for line in out.split('\n'):
    result = re.search(pathRegex, line)
    if result != None:
      filePath = result.group(1)
      if '.tar' not in filePath:
        continue
      file_list.append(result.group(1))
  return file_list


def clean_text(doc):
  """Clean the given text and returns a space separated string"""
  lowers = doc.lower()
  no_punctuation = ''.join([x for x in list(lowers) if x not in string.punctuation])
  tokens = nltk.word_tokenize(no_punctuation)
  filtered = [w for w in tokens if not w in stopwords.words('english')]
  stemmer = PorterStemmer()
  stemmed = []
  for item in filtered:
    stemmed.append(stemmer.stem(item))
  return stemmed


def download_extract_tar(s3FilePath):
  """For given tar s3FilePath copy to hdd, uncompress, return filePath to decompressed tar"""
  # Get tar file
  cmd = ['/usr/local/bin/s3cmd', 'get', '--skip-existing', s3FilePath, '--requester-pays', '/work']
  output = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
  print output
  # Extract file
  fileName = re.search(fileNameRegex, s3FilePath).group(1)
  print "Extracting " + fileName
  cmd = ['tar', 'xvf', fileName]
  subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
  folderName = re.search(folderNameRegex, fileName).group(1)
  # Set permissions so that files can be accessed
  cmd = ['chmod', '-R', '777', '/work/' + folderName]
  subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
  # Get doc list from newly extracted folder
  cmd = ['ls', '/work/' + folderName]
  output = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
  docFileNames = filter(lambda x: '.gz' in x, output[0].split('\n'))
  print "Extracting subfolders in: /work/" + folderName
  for gzip in docFileNames:
    docID = gzip.replace('.gz', '')
    cmd = ['mkdir', '/work/' + folderName + '/' + docID]
    subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
    cmd = ['tar', 'xvf', '/work/' + folderName + '/' + gzip, '-C', '/work/' + folderName + '/' + docID]
    subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
    cmd = ['chmod', '-R', '777', '/work/' + folderName + '/' + docID]
    subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
  cmd = ['rm', '/work/' + fileName]
  subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
  return '/work/' + folderName


def process_extracted_file_path(sc, sqlContext, extractedFilePath):
  textIDRDD = sc.wholeTextFiles("file://" + extractedFilePath + "/*/*.tex")
  def create_idDF(line):
    docName = re.search(arxivIDRegex, line[0]).group(1)
    searchResults = re.search(docURLRegex, docName)
    docURL = 'http://arxiv.org/abs/' + searchResults.group(1) + '/' + searchResults.group(2)
    return Row(**{"docName" : docName, "docURL" : docURL})
  idDF = sqlContext.createDataFrame(textIDRDD.map(create_idDF))
  content = map(lambda x: clean_text(x), textIDRDD.map(lambda x: x[1]).collect())
  contentDF = sqlContext.createDataFrame(sc.parallelize(content).map(lambda x: Row(**{"content" : x})))
  return idDF, contentDF


def load_data(sc, sqlContext):
  """Loads Data from arXiv, builds and saves models"""
  # Get file list and compare to old filelist get diff list
  s3FileList = load_s3_filelist()
  s3FileList = s3FileList[:2] # Remove for full run
  s3FileListDF = sqlContext.createDataFrame(sc.parallelize(s3FileList).map(lambda x: Row(**{"filePath" : x})))
  # Try to load old S3 File List, if it exists find diff, else just use new s3FileList
  try:
    oldS3FileListDF = sqlContext.read.load("/s3FileList")
    s3FileListNew = [x for x in map(lambda y: y.asDict()["filePath"], s3FileListDF.collect()) if x not in map(lambda y: y.asDict()["filePath"], oldS3FileListDF.collect())]
    if s3FileListNew == []:
      return "Nothing to update"
    update = True
  except Exception:
    s3FileListNew = map(lambda x: x.asDict()["filePath"], s3FileListDF.collect())
    update = False
  # For each s3 path, copy to hdd, uncompress, extract doc id and text from *.tex
  print "Processing " + s3FileListNew[0]
  extractedFilePath = download_extract_tar(s3FileListNew[0])
  idDF, contentDF = process_extracted_file_path(sc, sqlContext, extractedFilePath)
  # Save to idDF and contentDF to disk and cleanup
  try:
    contentDF.write.save("/content")
  except pyspark.sql.utils.AnalysisException:
    contentDF.write.mode("append").save("/content")
  try:
    idDF.write.save("/docID")
  except pyspark.sql.utils.AnalysisException:
    idDF.write.mode("append").save("/docID")
  print "Cleaning up folders on hard disk"
  cmd = ["rm", "-rf", extractedFilePath]
  subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
  for s3FilePath in s3FileListNew:
    if s3FilePath == s3FileListNew[0]:
      continue
    print "Processing " + s3FilePath
    extractedFilePath = download_extract_tar(s3FilePath)
    idDF, contentDF = process_extracted_file_path(sc, sqlContext, extractedFilePath)
    contentDF.write.mode("append").save("/content")
    idDF.write.mode("append").save("/docID")
    print "Cleaning up folders on hard disk"
    cmd = ["rm", "-rf", extractedFilePath]
    subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
  if update:
    s3FileListNewDF = sqlContext.createDataFrame(sc.parallelize(map(lambda x: Row(**{"filePath" : x}), s3FileListNew)))
    s3FileListNewDF.write.mode('append').save("/s3FileList")
  else:
    s3FileListDF.write.save("/s3FileList")
  return "Successfully processed: " + ', '.join(s3FileListNew)


def build_models(sc):
  # Once all tar files have been loaded into RDD
  # Create TF-IDF in spark
  print "Creating TF-IDF model"
  # Load the contentDF
  contentRDD = sqlContext.read.load("/content").rdd.map(lambda x: ' '.join(x[0]))
  hashingTF = HashingTF()
  tf = hashingTF.transform(contentRDD)
  idf = IDF(minDocFreq=2).fit(tf)
  tfidf = idf.transform(tf)
  # Create ALS model with documents as users and terms as products and tf-idf scores as ratings
  print "Creating ALS model"
  def create_training_data(line):
    docNum = line[1]
    termScores = np.column_stack((line[0].indices, line[0].values))
    result = []
    for row in termScores:
      result.append(Rating(docNum, row[0], row[1]))
    return result
  trainData = tfidf.zipWithIndex().flatMap(create_training_data)
  alsModel = ALS.train(trainData, 10, 10)
  # Save ALS model to HDFS
  alsModel.save(sc, "/ALS")
  return "Successfully built ALS and TF-IDF Models"


if __name__ == '__main__':
  import sys
  import getopt
  from pyspark import SparkContext
  from pyspark.sql import SQLContext

  # Setup Spark
  sc = SparkContext(master = "yarn", appName = "PaperPredictor")
  sqlContext = SQLContext(sc)

  sc.setLogLevel("FATAL")
  print load_data(sc, sqlContext)
  print build_models(sc)