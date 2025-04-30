# Heatmap: Correlation between numerical features (select only numeric columns).
plt.figure(figsize=(8, 6))
numeric_cols = df_sample.select_dtypes(include=np.number)
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()
