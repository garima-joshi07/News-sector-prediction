import pandas as pd
from tabulate import tabulate
data = pd.read_csv("created_df.csv")
print(data.head(3))
print(data.describe())
print(data.info())
print(data['label'].value_counts())