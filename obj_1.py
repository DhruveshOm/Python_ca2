
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
