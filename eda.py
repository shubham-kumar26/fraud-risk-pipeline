import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

engine = create_engine('postgresql://postgres:1234@localhost:5432/Sparkov')

# Pull data with hour extracted
query = """
SELECT category, amt, is_fraud, 
       EXTRACT(HOUR FROM trans_date_trans_time) AS hour_of_day
FROM transactions
"""
df = pd.read_sql(query, engine)

# Chart 1: Fraud rate by hour
hourly = df.groupby('hour_of_day')['is_fraud'].mean() * 100
plt.figure(figsize=(10,5))
hourly.plot(kind='bar', color='steelblue')
plt.title('Fraud Rate by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Fraud Rate (%)')
plt.tight_layout()
plt.savefig('fraud_by_hour.png')
plt.show()

print("Chart saved as fraud_by_hour.png")


# Chart 2: Fraud rate by category
category_fraud = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False) * 100
plt.figure(figsize=(10,5))
category_fraud.plot(kind='bar', color='indianred')
plt.title('Fraud Rate by Merchant Category')
plt.xlabel('Category')
plt.ylabel('Fraud Rate (%)')
plt.tight_layout()
plt.savefig('fraud_by_category.png')
plt.show()

print("Chart saved as fraud_by_category.png")





# Chart 3: Amount distribution - fraud vs non-fraud
plt.figure(figsize=(10,5))
sns.boxplot(x='is_fraud', y='amt', data=df[df['amt'] < 2000])  # capped for readability
plt.title('Transaction Amount: Fraud vs Non-Fraud')
plt.xlabel('Is Fraud')
plt.ylabel('Amount ($)')
plt.tight_layout()
plt.savefig('amount_distribution.png')

print("Chart saved as amount_distribution.png")
