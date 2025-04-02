import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("vgchartz-2024.csv")  # Replace with actual file path

# Extract a random sample of 10,000 rows
df_sample = df.sample(n=10000, random_state=42).reset_index(drop=True)



# Display the first 5 rows to understand dataset structure
print(df_sample.head())

# Check data types and convert date column
df_sample['release_date'] = pd.to_datetime(df_sample['release_date'], errors='coerce')
print(df_sample.dtypes)

# Function to filter games with high critic scores (Objective 3)
def filter_high_score(df, threshold=80):
    return df[df['critic_score'] > threshold]


# "Load the dataset, inspect its structure, and filter games with a critic score above a given threshold.

# Load the dataset and display the first 5 rows to understand its structure.
# Check the data types of all columns and convert them if necessary (e.g., dates to datetime format).
# Write a function to filter and display only the games with a critic score above a given threshold.