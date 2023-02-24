# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Regression Recipe Databricks Notebook
# MAGIC This notebook runs the MLflow Regression Recipe on Databricks and inspects its results.
# MAGIC 
# MAGIC For more information about the MLflow Regression Recipe, including usage examples,
# MAGIC see the [Regression Recipe overview documentation](https://mlflow.org/docs/latest/recipes.html#regression-recipe)
# MAGIC and the [Regression Recipe API documentation](https://mlflow.org/docs/latest/python_api/mlflow.recipes.html#module-mlflow.recipes.regression.v1.recipe).

# COMMAND ----------

# MAGIC %pip install git+https://github.com/mlflow/mlflow@master

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %md ### Start with a recipe:

# COMMAND ----------

from mlflow.recipes import Recipe

r = Recipe(profile="databricks")

# COMMAND ----------

# MAGIC %md ### Inspect recipe DAG:

# COMMAND ----------

r.inspect()

# COMMAND ----------

# MAGIC %md ### Ingest the dataset:

# COMMAND ----------

r.run("ingest")

# COMMAND ----------

# MAGIC %md ### Split the dataset into train, validation and test:

# COMMAND ----------

r.run("split")

# COMMAND ----------

training_data = r.get_artifact("training_data")
training_data.describe()

# COMMAND ----------

r.run("transform")

# COMMAND ----------

# MAGIC %md ### Train the model:

# COMMAND ----------

r.run("train")

# COMMAND ----------

trained_model = r.get_artifact("model")
print(trained_model)

# COMMAND ----------

run_data = r.get_artifact("run").data
run_id = run_data.tags["mlflow.rootRunId"]

# COMMAND ----------

# MAGIC %md ### Evaluate the model:

# COMMAND ----------

r.run("evaluate")

# COMMAND ----------

# MAGIC %md ### Register the model:

# COMMAND ----------

r.run("register")

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()

current_mv = r.get_artifact("registered_model_version")
trained_model = r.get_artifact("model")
training_data = r.get_artifact("training_data")
run_id = r.get_artifact("run").data.tags["mlflow.rootRunId"]
with mlflow.start_run(run_id = run_id):
  mlflow.pyfunc.log_model(f"{current_mv.name}_wrapper_model_with_input_examples", 
                          python_model=trained_model.unwrap_python_model(), 
                          signature=trained_model.metadata.signature,
                          input_example=training_data.drop("target", axis=1).loc[:3,:],##label column name has to be manually entered
                          pip_requirements= [ "mlflow>=2.1",
                                              "cloudpickle==2.0.0",
                                              "scikit-learn==1.1.3"
                                             ],
                          registered_model_name=current_mv.name) 

# COMMAND ----------

mlflow.pyfunc.load_model(f"""models:/{current_mv.name}/{int(current_mv.version)+1}""")

# COMMAND ----------

# MAGIC %md ### Register a new wrapper model with post-processing

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

# import json

# sample_input_json = {"dataframe_split": {"index": [0],
#   "columns": ["f1", "f2", "target", "date", "timestamp"],
#   "data": [[779,
#     0.5741065092576907,
#     1558.2543203319392,
#     "2020-12-31",
#     "2020-12-31 02:40:11"]]
# }
# }

# json.dumps(sample_input_json)


# COMMAND ----------


