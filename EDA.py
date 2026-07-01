import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

file_path = 'dataset.csv'
if not os.path.exists(file_path):
    print(f"Error : {file_path} is not found")
    exit

print("Loading the dataset:")

df = pd.read_csv(file_path)
print(f"Successfully Loaded. Rows:{df.shape[0]},Columns:{df.shape[1]}\n")
print(f"Displaying first 5 rows\n {df.head()}\n")
print(f"Displaying last 5 rows\n {df.tail()}\n")
print(f"Describing the dataset for understanding\n {df.describe()}\n")
print(f"Understanding the Datatypes of the features of the Dataset\n {df.info}\n")

print(f"Check Any Null Values in Dataset:\n{df.isnull().sum()}")

plt.figure(figsize=(7,4))
plt.hist()

