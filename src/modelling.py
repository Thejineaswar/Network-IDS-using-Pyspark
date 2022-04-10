#IMPORTS & DECLARATIONS
from pyspark.sql import SparkSession
import numpy as np
import time
import pandas as pd
from pyspark.sql.functions import col, lit, udf, when,count,isnan,first, col,row_number

from pyspark.sql.types import DoubleType

from pyspark.sql.functions import max as sparkMax
from pyspark.sql.functions import min as sparkMin
from pyspark.sql.window import Window

from pyspark.ml.feature import Imputer,VectorAssembler,MinMaxScaler,StringIndexer
from pyspark.ml.classification import RandomForestClassifier,DecisionTreeClassifier,MultilayerPerceptronClassifier,NaiveBayes

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from sklearn.metrics import confusion_matrix

#pip install pyspark=2.4.8
#pip install pandas
#pip install sklearn


PATHS = [
    'file:///mnt1/cicids2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
    'file:///mnt1/cicids2018/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
    # 'file:///mnt1/cicids2018/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
    # 'file:///mnt1/cicids2018/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
    # 'file:///mnt1/cicids2018/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
    # 'file:///mnt1/cicids2018/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv',
    # 'file:///mnt1/cicids2018/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv',
    # 'file:///mnt1/cicids2018/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv',
    # 'file:///mnt1/cicids2018/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv',
    # 'file:///mnt1/cicids2018/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv'
]


#################################CREATING SPARK SESSIONS AND READING THE FILE 
spark = SparkSession.builder\
        .master("local[*]")\
        .appName("ISAA_CIC_IDS_2018")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


print("Spark Session Created")
spark_df = spark.read.csv(PATHS,header = True, inferSchema = True)
print("CSV File Read")
seq_of_columns = spark_df.columns

# Using List comprehensions to create a list of columns of String DataType
string_columns = [i[0] for i in spark_df.dtypes if i[1]=='string']

# Using Set function to get non-string columns by subtracting one list from another.
non_string_columns = list(set(seq_of_columns) - set(string_columns))

########################### CODE TO REPLACE INFINITE VALUE
replace_infs_udf = udf(
    lambda x, v: float(v) if x and np.isinf(x) else x, DoubleType()
)
temp = spark_df.withColumn("Flow", replace_infs_udf(col("Flow Pkts/s"), lit(-100)))
des = temp.agg({"Flow":"max"}).collect()[0]

spark_df = spark_df.withColumn("Flow Pkts/s", replace_infs_udf(col("Flow Pkts/s"), lit(des[0])))
spark_df.agg({"Flow Pkts/s":"max"}).collect()[0]
print("Infinite Values replaced")

############## Dropping timestamp and imputing values
columns_to_drop = ['Timestamp']
spark_df = spark_df.drop(*columns_to_drop)
imputer = Imputer(inputCols=["Flow Byts/s"], outputCols=["Flow Byts/s"],strategy = "median")
imputer_mod = imputer.fit(spark_df)
spark_df = imputer_mod.transform(spark_df)

print("Dropped timestamp and imputing values")

#################### Stratified Split Feature Assembly

#Adding Row numbers
w = Window().orderBy(lit('Dst Port'))
df = spark_df.withColumn("row_num", row_number().over(w))
#Collecting all the distinct categories
h = df.select('Label').distinct().collect()
keys = {i[0] : 0.1 for i in h}
#Doing stratified sampling
sampled = df.sampleBy("Label", fractions=keys, seed=0)
#Collecting all the sampled rown numbers
qwert = sampled.select('row_num').collect()
#Remove the sampled row numbers
df = df.filter(~col("row_num").isin([i[0] for i in qwert]))

feature_assembler = VectorAssembler(inputCols = spark_df.columns[:-2],
                                    outputCol = "independent_features")
training = feature_assembler.transform(spark_df.na.drop())
validation = feature_assembler.transform(sampled.na.drop())

final_data = training.select("independent_features","Label")

scaler = MinMaxScaler(inputCol="independent_features",
                      outputCol="scaled_features")
scaler = scaler.fit(final_data)
training = scaler.transform(final_data)
validation = scaler.transform(validation.select("independent_features","Label"))

label_encoder = StringIndexer(inputCol = 'Label',outputCol = 'encoded_label')
label_encoder = label_encoder.fit(training)
train_data = label_encoder.transform(training)
valid_data = label_encoder.transform(validation)

################################### Modelling
print("Random Forest")
rf = RandomForestClassifier(featuresCol = 'independent_features', 
                            labelCol = 'encoded_label' )
start_time = time.time()
model = rf.fit(train_data)
print(f"Random Forest trained in {time.time() - start_time}")
rf_predictions = model.transform(valid_data)

print("Starting Decision tree")
gbt = DecisionTreeClassifier(labelCol="encoded_label", 
                    featuresCol="independent_features")
start_time = time.time()
gb_ = gbt.fit(train_data)
print(f" Decision tree trained in {start_time - time.time()}")
gb_predictions = gb_.transform(valid_data)
"""
print("Starting Naive Bayes")
start = time.time()
nb = NaiveBayes(smoothing=1.0, modelType="multinomial",labelCol="encoded_label", 
                    featuresCol="scaled_features")
nb_model = nb.fit(train_data)
print(f"Naive Bayes trained in {time.time() - start}")
nb_predictions = nb_model.transform(valid_data)
"""
print("Starting MLP")
NUM_LABELS = spark_df.select("Label").distinct().count()
layers = [len(feature_assembler.getInputCols()), 64, 32, NUM_LABELS]
classifier = MultilayerPerceptronClassifier(labelCol='encoded_label',
                                            featuresCol='scaled_features',
                                            maxIter=100,
                                            layers=layers,
                                            blockSize=128,
                                            seed=1234)
start = time.time()
classifier = classifier.fit(train_data)
print(f"MLP trained in {time.time() - start}")
mlp_predictions = classifier.transform(valid_data)

################################### EVALUATION
NAMES = [
        'Random Forest', 'Gradient Boosted Tree'
#'Naive Bayes','MLP'
]
DATA =[
      rf_predictions,gb_predictions
#,nb_predictions,mlp_predictions
]

METRICS = [
           'f1',
           'precisionByLabel',
           'recallByLabel',
           'accuracy'
]

def evaluate(data,metric_name):
  evaluator = MulticlassClassificationEvaluator(labelCol = "encoded_label",
                                              predictionCol = "prediction")
  evaluator.setMetricName(metric_name)
  print(f"################ {metric_name} #################")
  for i in range(len(data)):
    print(f"{NAMES[i]} : {evaluator.evaluate(data[i])}")
 
for i in METRICS:
    evaluate(DATA, i)
    
    
####################################### CONFUSION MATRIX
def get_cm_and_accuracy(predictions,name):
  y_true =  predictions.select(['encoded_label']).collect()
  y_pred =  predictions.select(['prediction']).collect()

  cm = confusion_matrix(y_true,y_pred)

  labels = [i for i in range(0,NUM_LABELS)]

  df_cm = pd.DataFrame(cm, index = labels,
                  columns = labels)

for i in range(len(NAMES)):
  get_cm_and_accuracy(DATA[i],NAMES[i])

