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
