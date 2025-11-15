import pandas as pd


df = pd.read_csv("data/train.csv")


print(df.head())
print(df.describe())
print(df.columns)



