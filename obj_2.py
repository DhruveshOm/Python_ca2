# -----------------------------------------
# Objective 2: top 10 best-selling games
# -----------------------------------------
# Sort the dataset based on total sales and find the top 10 best-selling games.
top10_sales = df_sample.sort_values(by='total_sales', ascending=False).head(10)
print("Top 10 Best-Selling Games:")
print(top10_sales[['title', 'total_sales']])
