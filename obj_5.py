# -----------------------------------------
# Objective 5: Identify outliers in total sales.
# -----------------------------------------
# Box plot:
plt.figure(figsize=(8, 5))
sns.boxplot(y=df_sample['total_sales'], color='violet')
plt.title('Outliers in Total Sales')
plt.tight_layout()
plt.show()
