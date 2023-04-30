from flask import Flask, request, jsonify
import json
import pyspark
from elasticsearch import Elasticsearch
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, MinMaxScaler, ChiSqSelector
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Définir le schéma de la  DataFrame
schema = StructType([
    StructField("Age", IntegerType(), True),
    StructField("Attrition", StringType(), True),
    StructField("BusinessTravel", StringType(), True),
    StructField("DailyRate", IntegerType(), True),
    StructField("Department", StringType(), True),
    StructField("DistanceFromHome", IntegerType(), True),
    StructField("Education", IntegerType(), True),
    StructField("EducationField", StringType(), True),
    StructField("EmployeeCount", IntegerType(), True),
    StructField("EmployeeNumber", IntegerType(), True),
    StructField("EnvironmentSatisfaction", IntegerType(), True),
    StructField("Gender", StringType(), True),
    StructField("HourlyRate", IntegerType(), True),
    StructField("JobInvolvement", IntegerType(), True),
    StructField("JobLevel", IntegerType(), True),
    StructField("JobRole", StringType(), True),
    StructField("JobSatisfaction", IntegerType(), True),
    StructField("MaritalStatus", StringType(), True),
    StructField("MonthlyIncome", IntegerType(), True),
    StructField("MonthlyRate", IntegerType(), True),
    StructField("NumCompaniesWorked", IntegerType(), True),
    StructField("Over18", StringType(), True),
    StructField("OverTime", StringType(), True),
    StructField("PercentSalaryHike", IntegerType(), True),
    StructField("PerformanceRating", IntegerType(), True),
    StructField("RelationshipSatisfaction", IntegerType(), True),
    StructField("StandardHours", IntegerType(), True),
    StructField("StockOptionLevel", IntegerType(), True),
    StructField("TotalWorkingYears", IntegerType(), True),
    StructField("TrainingTimesLastYear", IntegerType(), True),
    StructField("WorkLifeBalance", IntegerType(), True),
    StructField("YearsAtCompany", IntegerType(), True),
    StructField("YearsInCurrentRole", IntegerType(), True),
    StructField("YearsSinceLastPromotion", IntegerType(), True),
    StructField("YearsWithCurrManager", IntegerType(), True),
])

def preprocess_data(filename):
    data = spark.read.csv(filename, header=True, inferSchema=True)
    data_cleaned = data
    numerical_columns = ["Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]
    categorical_columns = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
    for column in numerical_columns:
        Q1 = data.approxQuantile(column, [0.25], 0.05)[0]
        Q3 = data.approxQuantile(column, [0.75], 0.05)[0]
        IQR = Q3 - Q1
        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR
        data_cleaned = data_cleaned.filter((col(column) >= lower_range) & (col(column) <= upper_range))
    stages = []
    for column in categorical_columns:
        indexer = StringIndexer(inputCol=column, outputCol=f"{column}Index")
        encoder = OneHotEncoder(inputCol=f"{column}Index", outputCol=f"{column}Vec")
        stages += [indexer, encoder]
    for column in numerical_columns:
        assembler = VectorAssembler(inputCols=[column], outputCol=f"{column}Vec")
        scaler = MinMaxScaler(inputCol=f"{column}Vec", outputCol=f"{column}Scaled")
        stages += [assembler, scaler]
    pipeline = Pipeline(stages=stages)
    data_prepared = pipeline.fit(data_cleaned).transform(data_cleaned)
    input_cols = [f"{col}Scaled" for col in numerical_columns] + [f"{col}Vec" for col in categorical_columns]
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    data_features = assembler.transform(data_prepared)
    indexer = StringIndexer(inputCol="Attrition", outputCol="label")
    data_features = indexer.fit(data_features).transform(data_features)
    selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    selected_data = selector.fit(data_features).transform(data_features)
    attrition_counts = selected_data.groupBy("label").count().collect()
    minority_count = min(attrition_counts, key=lambda x: x["count"])["count"]
    majority_count = max(attrition_counts, key=lambda x: x["count"])["count"]
    balancing_ratio = minority_count / majority_count
    majority_label = max(attrition_counts, key=lambda x: x["count"])["label"]
    minority_label = min(attrition_counts, key=lambda x: x["count"])["label"]
    majority_data = selected_data.filter(col("label") == majority_label)
    minority_data = selected_data.filter(col("label") == minority_label)
    majority_data_downsampled = majority_data.sample(withReplacement=False, fraction=balancing_ratio, seed=42)
    balanced_data = majority_data_downsampled.union(minority_data)

    return balanced_data, majority_data_downsampled, minority_data

def preprocess_input(data, majority_data_downsampled, minority_data):
    data_cleaned = data
    numerical_columns = ["Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]
    categorical_columns = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
    for column in numerical_columns:
        Q1 = data.approxQuantile(column, [0.25], 0.05)[0]
        Q3 = data.approxQuantile(column, [0.75], 0.05)[0]
        IQR = Q3 - Q1
        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR
        data_cleaned = data_cleaned.filter((col(column) >= lower_range) & (col(column) <= upper_range))
    stages = []
    for column in categorical_columns:
        indexer = StringIndexer(inputCol=column, outputCol=f"{column}Index")
        assembler = VectorAssembler(inputCols=[f"{column}Index"], outputCol=f"{column}Vec")
        stages += [indexer, assembler]
    for column in numerical_columns:
        assembler = VectorAssembler(inputCols=[column], outputCol=f"{column}Vec")
        scaler = MinMaxScaler(inputCol=f"{column}Vec", outputCol=f"{column}Scaled")
        stages += [assembler, scaler]
    pipeline = Pipeline(stages=stages)
    data_prepared = pipeline.fit(data_cleaned).transform(data_cleaned)
    input_cols = [f"{col}Scaled" for col in numerical_columns] + [f"{col}Vec" for col in categorical_columns]
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    data_features = assembler.transform(data_prepared)
    indexer = StringIndexer(inputCol="Attrition", outputCol="label")
    data_features = indexer.fit(data_features).transform(data_features)
    selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    selected_data = selector.fit(data_features).transform(data_features)
    balanced_data = majority_data_downsampled.union(minority_data)

    return balanced_data


spark = SparkSession.builder.appName("PAE - Prédiction Attrition Employés - Merouane Bennaceur V1.0").getOrCreate()
balanced_data, majority_data_downsampled, minority_data = preprocess_data("HR-Employee-Attrition.csv")
train_data, test_data = balanced_data.randomSplit([0.8, 0.2], seed=42)
lr = LogisticRegression(featuresCol="selectedFeatures", labelCol="label", maxIter=100)
best_lr_model = lr.fit(train_data)

app = Flask(__name__)
es = Elasticsearch("http://elasticsearch:9200")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    input_data = input_data["data"]
    # EXEMPLE
    # input_data = "34,No,Travel_Rarely,628,Research & Development,8,3,Medical,1,2068,2,Male,82,4,2,Laboratory Technician,3,Married,4404,10228,2,Y,No,12,3,1,80,0,6,3,4,4,3,1,2"
    input_list = input_data.split(",")
    input_list = [int(x) if x.isdigit() else x for x in input_list]

    # Créer un dataframe Spark à partir du schéma défini plus haut
    input_df = spark.createDataFrame([input_list], schema=schema)

    # Préparer les données d'entrée en utilisant le pipeline et extraire les features
    input_features = preprocess_input(input_df, majority_data_downsampled, minority_data)
    # Appliquer le modèle optimisé (par exemple, best_lr_model) aux données d'entrée
    prediction = best_lr_model.transform(input_features)
    # Convertir la prédiction en réponse JSON
    result = prediction.select("prediction", "probability").first().asDict()

    # Enregistrer les informations de surveillance dans Elasticsearch
    monitoring_data = {
        "timestamp": datetime.datetime.utcnow(),
        "input": input_data,
        "prediction": result["prediction"],
        "probability": result["probability"]
    }
    es.index(index="model_monitoring", body=monitoring_data)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)