import pandas as pd

df = pd.read_csv("cite_results.csv")
print(df.groupby(["model","variant", "precision", "verdict"]).size().unstack(fill_value=0))
print(df.groupby(["model"]).size())

