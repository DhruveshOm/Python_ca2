import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the dataset and display it 
# then Check the data types of all columns and convert them if necessary, 
# then write a function to filter and display only the games with a critic score above a given threshold.

df = pd.read_csv("vgchartz-2024.csv") 

sample = df.sample(n=10000, random_state=42).reset_index(drop=True)

print(sample.head())

sample['release_date'] = pd.to_datetime(sample['release_date'], errors='coerce')
print(sample.dtypes)

def score(df, threshold=80):
    return df[df['critic_score'] > threshold]
