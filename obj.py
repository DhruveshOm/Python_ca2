# ================================================================
# DATA SCIENCE ANALYSIS SCRIPT: VIDEO GAME SALES DATA
# ================================================================
# This script loads video game sales data, cleans it, and performs
# several analyses and visualizations, including descriptive and 
# inferential statistics, hypothesis testing, and data visualization.
# ================================================================

# -------------------------------
# Import all necessary libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Set Seaborn style for plots
sns.set(style="whitegrid")

# -------------------------------
# Load the dataset and take a sample
# -------------------------------
# Adjust the file path as needed. Using the uploaded file's path.

df = pd.read_csv("cleaned_data.csv")

# For consistency with your provided code, take a random sample of 10,000 rows.
# (Here, our dataset has 9,999 entries, so we'll work with the entire dataset.)
df_sample = df.copy()

# -----------------------------------------
# Objective 1: Load Data, Format Dates, and Filter by Critic Score
# -----------------------------------------
print("----- Objective 1: Data Overview and Filtering by Critic Score -----")
# Display the first 5 rows to understand the dataset structure.
print(df_sample.head())

# Convert 'release_date' column to datetime using dayfirst=True (the date format is dd/mm/yy)
df_sample['release_date'] = pd.to_datetime(df_sample['release_date'], dayfirst=True, errors='coerce')

# Check data types
print(df_sample.dtypes)

# Define a function to filter games with a critic score above a given threshold.
def filter_high_score(dataframe, threshold=80):
    return dataframe[dataframe['critic_score'] > threshold]

# Example usage: Filter and display games with critic score above 80.
high_score_games = filter_high_score(df_sample, threshold=80)
print(high_score_games.head())

# -----------------------------------------
# Objective 2: Categorize Games by Total Sales
# -----------------------------------------
print("----- Objective 2: Categorize Games by Total Sales -----")
# Define a function that categorizes total sales into "Low", "Medium", and "High".
def categorize_sales(sales):
    if sales > 5:
        return 'High'
    elif sales > 2:
        return 'Medium'
    else:
        return 'Low'

# Apply the categorization function to create a new column.
df_sample['sales_category'] = df_sample['total_sales'].apply(categorize_sales)
print(df_sample[['total_sales', 'sales_category']].head())

# -----------------------------------------
# Objective 3: Handle Missing Values, Create Sales Percentage, and Group by Genre
# -----------------------------------------
print("----- Objective 3: Handle Missing Values and Additional Metrics -----")
# Handling missing values: For simplicity, forward fill missing values.
df_sample.fillna(method='ffill', inplace=True)

# Create a new column that calculates the percentage contribution of each game's total sales.
df_sample['sales_percentage'] = (df_sample['total_sales'] / df_sample['total_sales'].sum()) * 100
print(df_sample[['total_sales', 'sales_percentage']].head())

# Group the data by "genre" and compute the average critic score for each genre.
genre_avg_score = df_sample.groupby('genre')['critic_score'].mean().reset_index()
print("Average Critic Score by Genre:")
print(genre_avg_score)

# Sort the dataset based on total sales and find the top 10 best-selling games.
top10_sales = df_sample.sort_values(by='total_sales', ascending=False).head(10)
print("Top 10 Best-Selling Games:")
print(top10_sales[['title', 'total_sales']])

# -----------------------------------------
# Objective 4: Visualizations – Bar Chart, Histogram, Scatter Plot, Pie Chart, and Heatmap
# -----------------------------------------
print("----- Objective 4: Visualizations -----")
# Bar chart: Total sales of top 5 publishers.
plt.figure(figsize=(10, 5))
top5_publishers = df_sample.groupby('publisher')['total_sales'].sum().nlargest(5)
top5_publishers.plot(kind='bar', color='skyblue')
plt.title('Top 5 Publishers by Total Sales')
plt.xlabel('Publisher')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# Histogram: Distribution of critic scores.
plt.figure(figsize=(8, 4))
sns.histplot(df_sample['critic_score'], bins=20, kde=True, color='green')
plt.title('Distribution of Critic Scores')
plt.xlabel('Critic Score')
plt.tight_layout()
plt.show()

# Scatter plot: Critic Score vs Total Sales.
plt.figure(figsize=(8, 5))
sns.scatterplot(x='critic_score', y='total_sales', data=df_sample, alpha=0.5)
plt.title('Critic Score vs Total Sales')
plt.xlabel('Critic Score')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# Pie chart: Sales distribution by region (NA, JP, PAL, Other).
plt.figure(figsize=(6, 6))
region_sales = df_sample[['na_sales', 'jp_sales', 'pal_sales', 'other_sales']].sum()
plt.pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%', startangle=140)
plt.title('Regional Sales Distribution')
plt.tight_layout()
plt.show()

# Heatmap: Correlation between numerical features (select only numeric columns).
plt.figure(figsize=(8, 6))
numeric_cols = df_sample.select_dtypes(include=np.number)
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# -----------------------------------------
# Objective 5: Advanced Analysis – Summary Statistics, Outliers, Correlation, Genre Distribution, and Chi-Square Test
# -----------------------------------------
print("----- Objective 5: Advanced Statistical Analysis -----")
# Summary statistics for all numerical columns.
print("Summary Statistics:")
print(df_sample.describe())

# Box plot: Identify outliers in total sales.
plt.figure(figsize=(8, 5))
sns.boxplot(y=df_sample['total_sales'], color='violet')
plt.title('Outliers in Total Sales')
plt.tight_layout()
plt.show()

# Correlation between critic score and total sales.
corr_value = df_sample['critic_score'].corr(df_sample['total_sales'])
print(f"Correlation between Critic Score and Total Sales: {corr_value:.2f}")

# Bar chart: Most common game genres.
plt.figure(figsize=(10, 5))
df_sample['genre'].value_counts().plot(kind='bar', color='orange')
plt.title('Most Common Game Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Chi-square test: Relationship between genre and publisher.
# To limit the table size, consider only the top 10 publishers.
top_publishers = df_sample['publisher'].value_counts().nlargest(10).index
filtered_df = df_sample[df_sample['publisher'].isin(top_publishers)]
contingency_table = pd.crosstab(filtered_df['genre'], filtered_df['publisher'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square test result: chi2 = {chi2:.2f}, p-value = {p:.5f}")

# -----------------------------------------
# New Objective 6: Test Normality of Critic Scores (Shapiro-Wilk Test)
# -----------------------------------------
print("----- New Objective 6: Normality Test for Critic Scores -----")
stat, p_value = shapiro(df_sample['critic_score'])
print(f"Shapiro-Wilk Test: Statistic = {stat:.3f}, p-value = {p_value:.5f}")
if p_value > 0.05:
    print("Data is likely normally distributed.")
else:
    print("Data is likely not normally distributed.")

# -----------------------------------------
# New Objective 7: Compare Total Sales between Action and Shooter Genres (t-test)
# -----------------------------------------
print("----- New Objective 7: t-Test between Action and Shooter Genres -----")
# Ensure the genres exist in your dataset
action_sales = df_sample[df_sample['genre'] == 'Action']['total_sales']
shooter_sales = df_sample[df_sample['genre'] == 'Shooter']['total_sales']
if len(action_sales) > 0 and len(shooter_sales) > 0:
    t_stat, p_val = ttest_ind(action_sales, shooter_sales, equal_var=False)
    print(f"T-Test: t-statistic = {t_stat:.3f}, p-value = {p_val:.5f}")
else:
    print("Not enough data for one or both genres for t-test.")

# -----------------------------------------
# New Objective 8: Check Multicollinearity using Variance Inflation Factor (VIF)
# -----------------------------------------
print("----- New Objective 8: Multicollinearity Check (VIF) -----")
# Select numeric columns related to sales.
numeric_data = df_sample[['na_sales', 'jp_sales', 'pal_sales', 'other_sales']]
X = add_constant(numeric_data)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# -----------------------------------------
# New Objective 9: Model Game Launch Frequency Per Year (Using a Bar Chart)
# -----------------------------------------
print("----- New Objective 9: Game Release Frequency Per Year -----")
# Extract the year from the release_date column.
df_sample['year'] = df_sample['release_date'].dt.year
year_counts = df_sample['year'].value_counts().sort_index()
plt.figure(figsize=(10, 5))
sns.barplot(x=year_counts.index, y=year_counts.values, palette="Blues_d")
plt.title('Number of Game Releases Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Games')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------------
# New Objective 10: A/B Testing Simulation – Compare Sales on Two Consoles (e.g., PS4 vs X360)
# -----------------------------------------
print("----- New Objective 10: A/B Testing Simulation (PS4 vs X360 Sales) -----")
ps4_sales = df_sample[df_sample['console'] == 'PS4']['total_sales']
x360_sales = df_sample[df_sample['console'] == 'X360']['total_sales']
if len(ps4_sales) > 0 and len(x360_sales) > 0:
    t_stat, p_val = ttest_ind(ps4_sales, x360_sales, equal_var=False)
    print(f"A/B Test (PS4 vs X360): t-statistic = {t_stat:.2f}, p-value = {p_val:.5f}")
else:
    print("Not enough data for one or both consoles for A/B test.")

# -------------------------------
# End of Script
# -------------------------------
print("----- Analysis Completed -----")
