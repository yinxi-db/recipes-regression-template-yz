def custom_split(df):
  """
  Mark rows of the ingested datasets to be split into training, validation, and test datasets.
  :param dataset: The dataset produced by the data ingestion step.
  :return: Series of strings with each element to be either 'TRAINING', 'VALIDATION' or 'TEST'
  """
  from pandas import Series
  from datetime import date
  
  splits = Series("TRAINING", index=range(len(df)))
  # 3rd quarter is validation data
  splits[df["date"] >= date(2021,1,1)] = "VALIDATION"
  # 4th quarter is testing data
  splits[df["date"] >= date(2022,1,1)] = "TEST"

  return splits

# def create_dataset_filter(dataset: DataFrame) -> Series(bool):
#   """
#   Mark rows of the split datasets to be additionally filtered. This function will be called on
#   the training, validation, and test datasets.
#   :param dataset: The {train,validation,test} dataset produced by the data splitting procedure.
#   :return: A Series indicating whether each row should be filtered
#   """

#   return (
#       (dataset["fare_amount"] > 0)
#       & (dataset["trip_distance"] < 400)
#       & (dataset["trip_distance"] > 0)
#       & (dataset["fare_amount"] < 1000)
#   ) | (~dataset.isna().any(axis=1))