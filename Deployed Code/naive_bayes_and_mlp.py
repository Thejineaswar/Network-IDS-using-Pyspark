

import time
import pandas as pd 
import numpy as np
import plotly.express as px 


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, when,count,isnan
from pyspark.sql.types import DoubleType

from pyspark.sql.functions import max as sparkMax
from pyspark.sql.functions import min as sparkMin
from pyspark.sql.functions import first, col

from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier,NaiveBayes


"""## Pyspark Setup

🚀 Pyspark Setup:
<ol> 
  <li> Java and spark installed </li>
  <li> Env Variables for the same are set </li>
  <li> Using findspark package initialise spark session </li>
"""


spark = SparkSession.builder\
        .master("local[*]")\
        .appName("ISAA_CIC_IDS_2018")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

sc=spark.sparkContext
sc.setLogLevel("ERROR")
print("Spark Session Created")


"""## Dataset Preprocessing

🚀 Dataset preprocessing:
<ol>
  <li> Dataset read and checked for Null and Infinity Values</li>
  <li> Infinity Values replaced with the max value in the column </li>
  <li> Null Values in columns handled using Imputer using median of column</li>
  <li> Categorical Column which includes the labels then encoded to numbers </li>
</ol>
"""

spark_df = spark.read.option("header","true").option("inferSchema",value=True).csv("s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",header  =True, inferSchema = True)
spark_df.printSchema()

#Checking for Null Values
spark_df.select([count(when(isnan(c), c)).alias(c) for c in spark_df.columns]).show()

seq_of_columns = spark_df.columns

# Using List comprehensions to create a list of columns of String DataType
string_columns = [i[0] for i in spark_df.dtypes if i[1]=='string']

# Using Set function to get non-string columns by subtracting one list from another.
non_string_columns = list(set(seq_of_columns) - set(string_columns))

print("Checking for the presence of infinite values")
df = spark_df.select(*[sparkMax(col(c)).alias(c) for c in non_string_columns],*[first(col(c),ignorenulls = True).alias(c) for c in string_columns])
df = df[[seq_of_columns]]
df.show()
print("Flow Pkts/s contains infinity")

print("Checking for the presence of negative infinite values")
df = spark_df.select(*[sparkMin(col(c)).alias(c) for c in non_string_columns],*[first(col(c),ignorenulls = True).alias(c) for c in string_columns])
df = df[[seq_of_columns]]
df.show() 
print("No column contains negative infinity")


replace_infs_udf = udf(
    lambda x, v: float(v) if x and np.isinf(x) else x, DoubleType()
)
temp = spark_df.withColumn("Flow", replace_infs_udf(col("Flow Pkts/s"), lit(-100)))

des = temp.agg({"Flow":"max"}).collect()[0]


spark_df = spark_df.withColumn("Flow Pkts/s", replace_infs_udf(col("Flow Pkts/s"), lit(4000000.0)))
spark_df.agg({"Flow Pkts/s":"max"}).collect()[0]
print("Infinite Values replaced")

columns_to_drop = ['Timestamp']
spark_df = spark_df.drop(*columns_to_drop)



spark_df = spark_df.fillna(value=0)

imputer = Imputer(inputCols=["Flow Byts/s"], outputCols=["Flow Byts/s"],strategy = "median")
imputer_mod = imputer.fit(spark_df)
spark_df = imputer_mod.transform(spark_df)

"""## Modelling

🚀 Modelling:
<ol>
<li> All independent features compiled using vector assembler </li> 
<li> The data is scaled in the range of 0 and 1 </li>
<li> Data split into train-test in 80/20 ratio</li>
<li> Random Forest, Decision Tree and Naive Bayes classification models trained </li>
<li> Test predictions evaluated using Accuracy </li>
</ol>
"""


feature_assembler = VectorAssembler(inputCols = spark_df.columns[:-1],
                                    outputCol = "independent_features")
training = feature_assembler.transform(spark_df)



final_data = training.select("independent_features","Label")

scaler = MinMaxScaler(inputCol="independent_features", outputCol="scaled_features")
final_data = scaler.fit(final_data).transform(final_data)

label_encoder = StringIndexer(inputCol = 'Label',outputCol = 'encoded_label')
final_data = label_encoder.fit(final_data).transform(final_data)




train_data, valid_data = final_data.randomSplit([0.8,0.2])


start_time=time.time()
nb = NaiveBayes(smoothing=1.0, modelType="multinomial",labelCol="encoded_label", 
                     featuresCol="scaled_features")
                     
nb_model = nb.fit(train_data)
nb_predictions = nb_model.transform(valid_data)
print(f"Time taken by navie_bayes {time.time()-start_time}")


layers = [len(feature_assembler.getInputCols()), 4, 2, 3]


start_time = time.time()

classifier = MultilayerPerceptronClassifier(labelCol='encoded_label',
                                            featuresCol='scaled_features',
                                            maxIter=100,
                                            layers=layers,
                                            blockSize=128,
                                            seed=1234)
classifier = classifier.fit(train_data)
mlp_predictions = classifier.transform(valid_data)

print(f"Time taken by MLP {time.time() - start_time}")
