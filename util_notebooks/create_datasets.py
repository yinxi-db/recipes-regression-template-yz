# Databricks notebook source
# MAGIC %md
# MAGIC Create a dummy dataset for MLflow recipe dogfooding

# COMMAND ----------

# MAGIC %fs ls /tmp/yz/recipe_ingest_data

# COMMAND ----------

dbutils.widgets.text("ingest_table_name","yz.recipe_ingest_data")
dbutils.widgets.text("ingest_table_path","dbfs:/tmp/yz/recipe_ingest_data")

# COMMAND ----------

from pyspark.sql.functions import col, rand, lit, date_sub, unix_timestamp, to_timestamp, current_date, current_timestamp
df = (spark
      .range(10000)
      .withColumnRenamed("id","f1")
      .withColumn("f2", rand(seed=123))
      .withColumn("target", 2*col("f1")+rand(seed=1234))
      .withColumn("date_today", current_date())
      .withColumn("date", date_sub("date_today", col("f1").cast("int")))
      .withColumn("time_now", current_timestamp())
      .withColumn("timestamp", to_timestamp(unix_timestamp("time_now") - col("f1")*60*60*24))
      .drop("date_today","time_now")
     )
display(df)

# COMMAND ----------

df.write.mode("overWrite").option("overwriteSchema", "true").saveAsTable(dbutils.widgets.get("ingest_table_name"))

# COMMAND ----------

df.write.mode("overWrite").option("overwriteSchema", "true").save(dbutils.widgets.get("ingest_table_path"))

# COMMAND ----------


