import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("vgchartz-2024.csv")  
df_sample = df.sample(n=10000, random_state=42).reset_index(drop=True)



# Implement a loop to categorize games based on their total sales into "Low," "Medium," and "High" sales groups.
def categorize_sales(sales):
    if sales > 5:
        return 'High'
    elif sales > 2:
        return 'Medium'
    else:
        return 'Low'

df_sample['sales_category'] = df_sample['total_sales'].apply(categorize_sales)




