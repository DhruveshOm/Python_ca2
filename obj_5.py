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


# Compute and display summary statistics (mean, median, standard deviation) for all numerical columns.

# Identify and visualize outliers in the total sales column using a box plot.

# Calculate the correlation between critic scores and total sales to determine if higher scores lead to more sales.

# Find the most common game genres and plot their distribution.

# Perform a chi-square test to check if there is a significant relationship between game genre and publisher.








# You said:
# write code also and make sure that i should extract 10k data from 60k


# ChatGPT said:
# I'll provide Python code that:
# Loads the dataset.
# Extracts a random 10,000-row sample from the 60,000-row dataset.
# Performs all 20 objectives using NumPy, Pandas, Matplotlib, and Seaborn.
# Here's the complete implementation:
# Data Science Objectives
# This script efficiently extracts a 10,000-row random sample, processes it, and completes all 20 objectives using NumPy, Pandas, Matplotlib, Seaborn, and SciPy. Let me know if you need modifications or explanations! 🚀






# You said:
# write the objection in comment and then perform code


# ChatGPT said:
# Failed to edit
# Edited

# Answer in chat instead
# I've added comments describing each objective in the code. Let me know if you need modifications or further explanations! 🚀







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
