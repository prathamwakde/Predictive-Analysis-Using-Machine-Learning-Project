# Initialize SparkSession:
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, regexp_replace
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName('LinearRegression').getOrCreate()
spark
# Load and Prepare Data:
df = spark.read.csv('movies.csv',header=True,inferSchema=True)

# Remove the '$' sign and cast the column to double
df = df.withColumn("Worldwide Gross", regexp_replace(col("Worldwide Gross"), "\\$", "").cast("double"))

df.show(5)

df.columns
# Feature Engineering
assembler = VectorAssembler(
    inputCols=["Audience score %","Profitability","Rotten Tomatoes %","Worldwide Gross"],
    outputCol="features")
output = assembler.transform(df)
output.select("features").show(5)
# Split Data
final_data = output.select("features","Year", "Profitability")
train_data , test_data = final_data.randomSplit([0.7,0.3])

train_data.describe().show()
test_data.describe().show()
#  Choose and Train the Model:
lr = LinearRegression(labelCol="Profitability")
lrModel = lr.fit(train_data)
lrModel
# Evaluate the Model:
test_results = lrModel.evaluate(test_data)
test_results.residuals.show()
# Stop SparkSession:
spark.stop()