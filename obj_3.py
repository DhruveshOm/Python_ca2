import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("vgchartz-2024.csv")  

# Extract a random sample of 10,000 rows
df_sample = df.sample(n=10000, random_state=42).reset_index(drop=True)


# Handling missing values (Objective 7)
df_sample.fillna(df_sample.mean(), inplace=True)

# New column for percentage contribution of sales (Objective 8)
df_sample['sales_percentage'] = (df_sample['total_sales'] / df_sample['total_sales'].sum()) * 100

# Group by genre and compute average critic score (Objective 9)
avg_critic_by_genre = df_sample.groupby('genre')['critic_score'].mean()

# Top 10 best-selling games (Objective 10)
top_10_games = df_sample.sort_values(by='total_sales', ascending=False).head(10)

