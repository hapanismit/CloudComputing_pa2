from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

import pyspark.sql.functions as func
import pyspark

conf = SparkConf().setAppName("Wine Quality Prediction").setMaster("local[4]")
sc = SparkContext(conf=conf)

spark = SparkSession.builder.getOrCreate()

#Read data from csv
data = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv("s3://pa2smit/TrainingDataset.csv")

print("\nPrinting Training Schema\n")
data.printSchema()
data.count()

featureColumns = [col for col in data.columns if col != '""""quality"""""']

assembler = VectorAssembler(inputCols=featureColumns, outputCol='values')
transformData = assembler.transform(data)

rf = RandomForestClassifier(featuresCol='values', labelCol='""""quality"""""',numTrees=100, maxBins=484, maxDepth=25, minInstancesPerNode=5, seed=34)
rfModel = rf.fit(transformData)

evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol="prediction", metricName="f1")
rfTrainingPredictions = rfModel.transform(transformData)

print("\nModel Training Completed ...\n")
print("\nRandom Forest f1 of traning data = %g\n" % evaluator.evaluate(rfTrainingPredictions))

rf.save("s3://pa2smit/wine_model.model")
