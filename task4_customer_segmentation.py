import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load Cleaned Dataset
df = pd.read_csv('cleaned_retail_data.csv')
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])

# 2. Engineer RFM Metrics (Recency, Frequency, Monetary)
snapshot_date = df['PurchaseDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'PurchaseDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'AmountSpent': 'sum'
}).rename(columns={
    'PurchaseDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'AmountSpent': 'Monetary'
})

print("=== 1. RFM Table Head ===")
print(rfm.head())

# 3. Customer Segmentation via K-Means Clustering
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Determine Persona Names by Spend Level
cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values()
spend_ranking = cluster_means.index

persona_map = {
    spend_ranking[2]: 'High-Value VIPs',
    spend_ranking[1]: 'Loyal Core Customers',
    spend_ranking[0]: 'At-Risk / Low Spenders'
}
rfm['Segment'] = rfm['Cluster'].map(persona_map)

print("\n=== 2. Segment Profiles (Averages) ===")
summary = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
summary['Customer_Count'] = rfm['Segment'].value_counts()
print(summary)

# 4. Actionable Marketing Recommendations
print("\n=== 3. Strategic Marketing Playbook ===")
print("• High-Value VIPs: Tiered loyalty perks, early product access, VIP support.")
print("• Loyal Core Customers: Volume discount bundles (e.g. Spend $100, Save $15).")
print("• At-Risk / Low Spenders: Re-activation drip sequence with tailored 20% discount codes.")

# 5. Customer Segmentation Visualization
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=rfm,
    x='Frequency',
    y='Monetary',
    hue='Segment',
    palette={'High-Value VIPs': '#2ca02c', 'Loyal Core Customers': '#1f77b4', 'At-Risk / Low Spenders': '#d62728'},
    s=90,
    alpha=0.85
)
plt.title('Customer Segmentation: Frequency vs Monetary Value', fontsize=13, fontweight='bold')
plt.xlabel('Purchase Frequency (Number of Transactions)', fontsize=11)
plt.ylabel('Total Spend / Monetary ($)', fontsize=11)
plt.legend(title='Customer Segment')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('customer_segments.png', dpi=300)
print("\n[Complete] Segmentation chart saved as 'customer_segments.png'.")
