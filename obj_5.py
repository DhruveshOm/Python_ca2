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


To perform statistical analysis and visualization on the video game dataset by computing summary statistics,
identifying outliers in total sales, analyzing the correlation between critic scores and sales,
exploring the distribution of game genres, and examining the relationship between genres and publishers using a chi-square test.


# Summary statistics (Objective 16)
print(df_sample.describe())


# Box plot: Outliers in total sales (Objective 17)
plt.figure(figsize=(8,5))
sns.boxplot(y=df_sample['total_sales'])
plt.title('Outliers in Total Sales')
plt.show()


# Correlation between critic score and total sales (Objective 18)
corr_value = df_sample['critic_score'].corr(df_sample['total_sales'])
print(f"Correlation between Critic Score and Total Sales: {corr_value:.2f}")


# Most common genres (Objective 19)
df_sample['genre'].value_counts().plot(kind='bar', color='orange')
plt.title('Most Common Game Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()

# Chi-square test: Relationship between genre and publisher (Objective 20)
contingency_table = pd.crosstab(df_sample['genre'], df_sample['publisher'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square test result: chi2={chi2:.2f}, p-value={p:.5f}")
