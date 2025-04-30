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
