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

# To clean, analyze, and visualize the video game dataset by handling missing values, computing regional sales contributions, 
# summary statistics, and correlations. The project also includes identifying top-selling games, analyzing genre and publisher relationships, 
# detecting outliers, 
# and creating visualizations such as bar charts, histograms, scatter plots, pie charts, box plots, and heatmaps.

# Bar chart: Total sales of top 5 publishers (Objective 11)
plt.figure(figsize=(10,5))
df_sample.groupby('publisher')['total_sales'].sum().nlargest(5).plot(kind='bar', color='skyblue')
plt.title('Top 5 Publishers by Total Sales')
plt.xlabel('Publisher')
plt.ylabel('Total Sales')
plt.show()

# Histogram: Distribution of critic scores (Objective 12)
plt.figure(figsize=(8,4))
sns.histplot(df_sample['critic_score'], bins=20, kde=True)
plt.title('Distribution of Critic Scores')
plt.show()

# Scatter plot: Critic Score vs Total Sales (Objective 13)
plt.figure(figsize=(8,5))
sns.scatterplot(x=df_sample['critic_score'], y=df_sample['total_sales'], alpha=0.5)
plt.title('Critic Score vs Total Sales')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales')
plt.show()

# Pie chart: Sales distribution by region (Objective 14)
region_sales = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']
df_sample[region_sales].sum().plot.pie(autopct='%1.1f%%', figsize=(6,6))
plt.title('Regional Sales Distribution')
plt.ylabel('')
plt.show()


# Heatmap: Correlation between numerical features (Objective 15)
plt.figure(figsize=(8,6))
sns.heatmap(df_sample.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

