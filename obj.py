# ================================================================
# DATA SCIENCE ANALYSIS SCRIPT: VIDEO GAME SALES DATA
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import  ttest_ind

# Set Seaborn style for plots
sns.set(style="whitegrid")
# Load the dataset and take a sample
df_sample = pd.read_csv("cleaned_data.csv")


# EDA :
# Display the first 5 rows to understand the dataset structure.
print(df_sample.head())
# Convert 'release_date' column to datetime using dayfirst=True (the date format is dd/mm/yy)
df_sample['release_date'] = pd.to_datetime(df_sample['release_date'], dayfirst=True, errors='coerce')
# Check data types
print(df_sample.dtypes)

# -----------------------------------------
# Objective 1: Categorize Games by Total Sales
# ----------------------------------------------
#  total sales into "Low", "Medium", and "High".
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
# Objective 2: top 10 best-selling games
# -----------------------------------------
# Sort the dataset based on total sales and find the top 10 best-selling games.
top10_sales = df_sample.sort_values(by='total_sales', ascending=False).head(10)
print("Top 10 Best-Selling Games:")
print(top10_sales[['title', 'total_sales']])

# -----------------------------------------
# Objective 3: Total sales of top 5 publishers.
# -----------------------------------------
# Bar chart:
plt.figure(figsize=(10, 5))
top5_publishers = df_sample.groupby('publisher')['total_sales'].sum().nlargest(5)
top5_publishers.plot(kind='bar', color='skyblue')
plt.title('Top 5 Publishers by Total Sales')
plt.xlabel('Publisher')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.show()

# --------------------------------------------------------------
# Objective 4: Sales distribution by region (NA, JP, PAL, Other)
# ---------------------------------------------------------------
# Pie chart: 
plt.figure(figsize=(6, 6))
region_sales = df_sample[['na_sales', 'jp_sales', 'pal_sales', 'other_sales']].sum()
plt.pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%', startangle=140)
plt.title('Regional Sales Distribution')
plt.tight_layout()
plt.show()

# -----------------------------------------
# Objective 5: Identify outliers in total sales.
# -----------------------------------------
# Box plot:
plt.figure(figsize=(8, 5))
sns.boxplot(y=df_sample['total_sales'], color='violet')
plt.title('Outliers in Total Sales')
plt.tight_layout()
plt.show()
# -----------------------------------------
# OBJECTIVE 6 : Most common game genres.
# -----------------------------------------
# Bar chart: 
plt.figure(figsize=(10, 5))
df_sample['genre'].value_counts().plot(kind='bar', color='orange')
plt.title('Most Common Game Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# -----------------------------------------
# Objective 7: Compare Total Sales between Action and Shooter Genres (t-test)
# -----------------------------------------

action_sales = df_sample[df_sample['genre'] == 'Action']['total_sales']
shooter_sales = df_sample[df_sample['genre'] == 'Shooter']['total_sales']
if len(action_sales) > 0 and len(shooter_sales) > 0:
    t_stat, p_val = ttest_ind(action_sales, shooter_sales, equal_var=False)
    print(f"T-Test: t-statistic = {t_stat:.3f}, p-value = {p_val:.5f}")
else:
    print("Not enough data for one or both genres for t-test.")

# -----------------------------------------
#   objective 8: Model Game Launch Frequency Per Year (Using a Bar Chart)
# -----------------------------------------
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


