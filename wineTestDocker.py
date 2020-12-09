from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
import sys
import pyspark.sql.functions as func
import pyspark

conf = SparkConf().setAppName("Wine Quality Prediction").setMaster("local[1]")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

#loading trained model
rf = RandomForestClassifier.load("wine_model.model")

#Read data from csv
#data = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv("s3://pa2smit/ValidationDataset.csv")
data = spark.read.csv('ValidationDataset.csv',header='true', inferSchema='true', sep=';')

featureColumns = [col for col in data.columns if col != '""""quality"""""']
assembler = VectorAssembler(inputCols=featureColumns, outputCol='values')

rfPipe = Pipeline(stages=[assembler, rf])

fitData = rfPipe.fit(data)
transformedData = fitData.transform(data)
transformedData = transformedData.withColumn("prediction", func.round(transformedData['prediction']))
transformedData = transformedData.withColumn('""""quality"""""', transformedData['""""quality"""""'].cast('double')).withColumnRenamed('""""quality"""""', "label")

results = transformedData.select(['prediction', 'label'])
predictionAndLabels = results.rdd

metrics = MulticlassMetrics(predictionAndLabels)

#Overall statistics
precision = metrics.precision(5)
recall = metrics.recall(5)
f1Score = metrics.fMeasure(5)
print("Summary Statistics")
print("=======================")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)
print("=======================")
