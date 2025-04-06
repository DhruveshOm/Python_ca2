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


# Display the first 5 rows
print(df_sample.head())

# Check data types and convert date column
df_sample['release_date'] = pd.to_datetime(df_sample['release_date'], errors='coerce')
print(df_sample.dtypes)

# Function to filter games with high critic scores
def filter_high_score(df, threshold=80):
    return df[df['critic_score'] > threshold]

# Categorizing games based on total sales
def categorize_sales(sales):
    if sales > 5:
        return 'High'
    elif sales > 2:
        return 'Medium'
    else:
        return 'Low'

df_sample['sales_category'] = df_sample['total_sales'].apply(categorize_sales)

# Handling missing values
df_sample.fillna(df_sample.mean(), inplace=True)

# New column for percentage contribution of sales
df_sample['sales_percentage'] = (df_sample['total_sales'] / df_sample['total_sales'].sum()) * 100

# Group by genre and compute average critic score
avg_critic_by_genre = df_sample.groupby('genre')['critic_score'].mean()

# Top 10 best-selling games
top_10_games = df_sample.sort_values(by='total_sales', ascending=False).head(10)

# Bar chart: Total sales of top 5 publishers
plt.figure(figsize=(10,5))
df_sample.groupby('publisher')['total_sales'].sum().nlargest(5).plot(kind='bar', color='skyblue')
plt.title('Top 5 Publishers by Total Sales')
plt.xlabel('Publisher')
plt.ylabel('Total Sales')
plt.show()

# Histogram: Distribution of critic scores
plt.figure(figsize=(8,4))
sns.histplot(df_sample['critic_score'], bins=20, kde=True)
plt.title('Distribution of Critic Scores')
plt.show()


# Scatter plot: Critic Score vs Total Sales
plt.figure(figsize=(8,5))
sns.scatterplot(x=df_sample['critic_score'], y=df_sample['total_sales'], alpha=0.5)
plt.title('Critic Score vs Total Sales')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales')
plt.show()

# Pie chart: Sales distribution by region
region_sales = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']
df_sample[region_sales].sum().plot.pie(autopct='%1.1f%%', figsize=(6,6))
plt.title('Regional Sales Distribution')
plt.ylabel('')
plt.show()

# Heatmap: Correlation between numerical features
plt.figure(figsize=(8,6))
sns.heatmap(df_sample.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Summary statistics
print(df_sample.describe())

# Box plot: Outliers in total sales
plt.figure(figsize=(8,5))
sns.boxplot(y=df_sample['total_sales'])
plt.title('Outliers in Total Sales')
plt.show()

# Correlation between critic score and total sales
corr_value = df_sample['critic_score'].corr(df_sample['total_sales'])
print(f"Correlation between Critic Score and Total Sales: {corr_value:.2f}")

# Most common genres
df_sample['genre'].value_counts().plot(kind='bar', color='orange')
plt.title('Most Common Game Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()

# Chi-square test: Relationship between genre and publisher
contingency_table = pd.crosstab(df_sample['genre'], df_sample['publisher'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square test result: chi2={chi2:.2f}, p-value={p:.5f}")




# https://mavenanalytics.io/data-playground?order=number_of_records%2Cdesc&page=4&pageSize=5
