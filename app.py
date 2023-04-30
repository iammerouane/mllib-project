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



# Création de l'application Spark avec le nom PAE - Prédiction Attrition Employés - Merouane Bennaceur V1.0
spark = SparkSession.builder.appName("PAE - Prédiction Attrition Employés - Merouane Bennaceur V1.0").getOrCreate()

# Charger les données
data = spark.read.csv("HR-Employee-Attrition.csv", header=True, inferSchema=True)

# Afficher les 10 premières lignes
print("\n\n")
print("Affichage des 10 premières lignes")
data.show(10)

# Afficher le schéma
print("\n\n")
print("Affichage du schéma")
data.printSchema()

# Statistiques descriptives
print("\n\n")
print("Statistiques descriptives")
data.describe().show()
print("\n\n")
data_summary = data.describe().toPandas()
print(data_summary)

# Vérification des valeurs manquantes
# Identifiecation des colonnes avec des valeurs manquantes
print("\n\n")
print("Valeurs manquantes par colonnes")
data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()


# Analyse des distributions de variables
# Exemple avec age
print("\n\n")
print("Histogramme de distribution de la variable age")
col_name = "age"
col_data = data.select(col_name).toPandas()
sns.histplot(col_data[col_name], kde=True)
plt.show()


# Examiner les corrélations entre les variables et visualiser les relations
# Création d'un vecteur avec les colonnes numériques
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=["Age", "DailyRate"], outputCol=vector_col)
data_vector = assembler.transform(data).select(vector_col)

# Calculer la matrice de corrélation
matrix = Correlation.corr(data_vector, vector_col).collect()[0][0]
correlation_matrix = matrix.toArray().tolist()
print("\n\n")
print("Matrice de corrélation entre Age et DailyRate")
print(correlation_matrix)

# Exemple pour un nuage de points entre deux variables numériques (Age, DailyRate)
x_col = "Age"
y_col = "DailyRate"
scatter_data = data.select(x_col, y_col).toPandas()
sns.scatterplot(data=scatter_data, x=x_col, y=y_col)
print("\n\n")
print("Nuage de points de corrélation entre Age et DailyRate")
plt.show()

# On va supprimer les valeurs aberrantes les colonnes numériques en utilisant l'intervalle interquartile (IQR).

# Liste des colonnes numériques
numerical_columns = ["Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]

data_cleaned = data

for column in numerical_columns:
    Q1 = data.approxQuantile(column, [0.25], 0.05)[0]
    Q3 = data.approxQuantile(column, [0.75], 0.05)[0]
    IQR = Q3 - Q1
    lower_range = Q1 - 1.5 * IQR
    upper_range = Q3 + 1.5 * IQR

    data_cleaned = data_cleaned.filter((col(column) >= lower_range) & (col(column) <= upper_range))

# Affichage dataset nettoyé
print("\n\n")
print("Affichage dataset nettoyé")
data_clean.show()

# Normalisation de toutes les colonnes numériques et transformation des colonnes discrettes en variables numériques (one-hot encoding).

# Liste des colonnes discrétes
categorical_columns = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]

# Stages pour le pipeline
stages = []

# One-hot encoding pour les colonnes catégorielles
for column in categorical_columns:
    indexer = StringIndexer(inputCol=column, outputCol=f"{column}Index")
    encoder = OneHotEncoder(inputCol=f"{column}Index", outputCol=f"{column}Vec")
    stages += [indexer, encoder]

# VectorAssembler et MinMaxScaler pour les colonnes numériques
for column in numerical_columns:
    assembler = VectorAssembler(inputCols=[column], outputCol=f"{column}Vec")
    scaler = MinMaxScaler(inputCol=f"{column}Vec", outputCol=f"{column}Scaled")
    stages += [assembler, scaler]

# Créer et appliquer un pipeline
pipeline = Pipeline(stages=stages)
data_prepared = pipeline.fit(data_clean).transform(data_clean)

# Affichage dataset préparé
print("\n\n")
print("Affichage dataset préparé")
data_prepared.show()

# Utilisation méthode Chi-Squared pour sélectionner les fonctionnalités les plus pertinentes pour le modèle. 
# pour sélectionner les 10 fonctionnalités les + importantes parmi les colonnes discrétes et numériques prétraitées.

# Combinez toutes les colonnes d'entités en un seul vecteur
input_cols = [f"{col}Scaled" for col in numerical_columns] + [f"{col}Vec" for col in categorical_columns]
assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
data_features = assembler.transform(data_prepared)

# Indexez la colonne Attrition (cible)
indexer = StringIndexer(inputCol="Attrition", outputCol="label")
data_features = indexer.fit(data_features).transform(data_features)

# Utilisation ChiSqSelector pour sélectionner les fonctionnalités les plus pertinentes
selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
selected_data = selector.fit(data_features).transform(data_features)
selected_data = selected_data.select("features","selectedFeatures", "label")

# Affichage dataset avec les fonctionnalités pertinentes
print("\n\n")
print("Affichage dataset avec les fonctionnalités pertinentes")
selected_data.show()

# Calculez le taux d'équilibrage
attrition_counts = selected_data.groupBy("label").count().collect()
minority_count = min(attrition_counts, key=lambda x: x["count"])["count"]
majority_count = max(attrition_counts, key=lambda x: x["count"])["count"]
balancing_ratio = minority_count / majority_count

# Équilibrez les données
majority_label = max(attrition_counts, key=lambda x: x["count"])["label"]
minority_label = min(attrition_counts, key=lambda x: x["count"])["label"]

majority_data = selected_data.filter(col("label") == majority_label)
minority_data = selected_data.filter(col("label") == minority_label)

majority_data_downsampled = majority_data.sample(withReplacement=False, fraction=balancing_ratio, seed=42)
balanced_data = majority_data_downsampled.union(minority_data)

print("\n\n")
print("Affichage dataset équilibré")
balanced_data.show()

train_data, test_data = balanced_data.randomSplit([0.8, 0.2], seed=42)

# Random Forest
rf = RandomForestClassifier(featuresCol="selectedFeatures", labelCol="label", numTrees=100)
rf_model = rf.fit(train_data)


# Gradient Boosting
gbt = GBTClassifier(featuresCol="selectedFeatures", labelCol="label", maxIter=100)
gbt_model = gbt.fit(train_data)


# Logistic Regression:
lr = LogisticRegression(featuresCol="selectedFeatures", labelCol="label", maxIter=100)
lr_model = lr.fit(train_data)

# Decision Tree:
dt = DecisionTreeClassifier(featuresCol="selectedFeatures", labelCol="label")
dt_model = dt.fit(train_data)

# Fonction pour évaluer les modèles
def evaluate_model(model, test_data):
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator()
    roc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
    predictionAndLabels = predictions.select("prediction", "label").rdd.map(lambda p: (float(p[0]), float(p[1])))
    metrics = MulticlassMetrics(predictionAndLabels)
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1_score = metrics.fMeasure(1.0)
    return roc, precision, recall, f1_score

# Évaluation des modèles
models = [("Random Forest", rf_model), ("Gradient Boosting", gbt_model), ("Logistic Regression", lr_model), ("Decision Tree", dt_model)]

for model_name, model in models:
    roc, precision, recall, f1_score = evaluate_model(model, test_data)
    print(f"{model_name}:")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1 Score: {f1_score}")
    print(f"  ROC: {roc}\n")
    
    
# Pour optimiser les modèles on peut utiliser la validation croisée avec un ensemble de paramètres à tester. 
# Voici un exemple avec la régression logistique car c'est elle qui a obtenu les meilleurs évaluations

# Optimisation du modèle avec validation croisée et recherche sur grille pour la régression logistique
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
    .addGrid(lr.fitIntercept, [True, False]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

cross_val = CrossValidator(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=BinaryClassificationEvaluator(),
                           numFolds=5)

cv_model = cross_val.fit(train_data)
best_lr_model = cv_model.bestModel

# Évaluer le modèle optimisé
roc, precision, recall, f1_score = evaluate_model(best_lr_model, test_data)
print("Optimized Logistic Regression:")
print(f"  Precision: {precision}")
print(f"  Recall: {recall}")
print(f"  F1 Score: {f1_score}\n")
print(f"  Area Under ROC: {roc}\n")




app = Flask(__name__)
es = Elasticsearch("http://elasticsearch:9200")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    # Convertir les données d'entrée en DataFrame Spark
    input_df = spark.createDataFrame([input_data])
    # Préparer les données d'entrée en utilisant le pipeline
    input_prepared = pipeline.fit(input_df).transform(input_df)
    input_features = assembler.transform(input_prepared)
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