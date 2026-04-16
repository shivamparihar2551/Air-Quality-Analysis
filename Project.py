import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dynamic_supply_chain_logistics_dataset.csv")

# Display basic info
print(df.head())
print(df.info())
print(df.describe())


print("Shape:", df.shape)
print("Columns:", df.columns)

print(df.isnull().sum())
