# Fraud Risk Pipeline — SQL Analytics + ML Classification

An end-to-end fraud detection project on credit card transaction data — combining SQL-based exploratory analysis, Python feature engineering and machine learning, and a Power BI dashboard.

**Live Dashboard:** [View on Power BI](https://app.powerbi.com/links/AvMnaDKVLh?ctid=bc5b2879-3fac-469a-b8c4-994705bc09d7&pbi_source=linkShare)

---

## Problem Statement

Financial institutions process millions of transactions daily, and a small fraction are fraudulent. Manually reviewing every transaction isn't feasible — this project builds a system to flag likely-fraudulent transactions automatically, using historical transaction patterns.

## Dataset

**Source:** [Sparkov Credit Card Transactions (Kaggle)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

- 1,296,675 transactions (Jan 2019 – Dec 2020)
- 7,506 labeled fraud cases (~0.58% fraud rate — highly imbalanced)
- Fields include transaction amount, merchant category, customer/merchant location, demographics, and timestamp

I chose Sparkov over the more commonly-used PaySim/mlg-ulb datasets because it includes richer, more interpretable features (merchant category, location, demographics) that support meaningful SQL-based business analysis rather than working with anonymized PCA columns.

## Tech Stack

- **PostgreSQL** — data storage and exploratory SQL analysis
- **Python** (Pandas, scikit-learn, XGBoost) — feature engineering and ML modeling
- **Power BI** — interactive dashboard

---

## 1. SQL Exploratory Analysis

All queries in [`fraud_queries.sql`](./fraud_queries.sql). Key findings:

| Finding | Result |
|---|---|
| Fraud rate by category | Online categories (`shopping_net` 1.76%, `misc_net` 1.45%) show notably higher fraud than in-person equivalents |
| Fraud rate by age group | Customers 60+ have the highest fraud rate (0.75%) |
| Distance (customer–merchant) | No meaningful difference between fraud and non-fraud (47.30 vs 47.39 miles) — honest negative result |
| **Fraud rate by hour** | **Strongest signal** — 22:00–23:00 shows ~2.8% fraud rate vs ~0.1% during the day (20–30x higher) |
| Spending spike detection (LAG/CTE) | Large sudden jumps in spend do NOT reliably indicate fraud — legitimate large purchases look identical |
| Transaction amount | Fraud averages $531 vs $68 for legitimate transactions, and never exceeds ~$1,376 |

This analysis used window functions (`RANK`, `LAG`), CTEs, and aggregate queries — including negative results, since testing and ruling out weak hypotheses (distance, simple spend spikes) is as important as finding strong ones.

## 2. Python EDA

Charts built with Pandas/Matplotlib/Seaborn, connected directly to PostgreSQL via SQLAlchemy.

![Fraud rate by hour](./fraud_by_hour.png)
![Fraud rate by category](./fraud_by_category.png)
![Amount distribution](./amount_distribution.png)

## 3. Feature Engineering

From raw transaction data, engineered:
- `hour_of_day` — extracted from timestamp
- `age` — calculated from date of birth
- `distance_miles` — Haversine distance between customer and merchant coordinates
- `category_encoded`, `gender_encoded` — categorical encoding for modeling

## 4. Model Comparison

Trained on an 80/20 stratified split, handling class imbalance via `class_weight`/`scale_pos_weight`:

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.08 | 0.76 | 0.15 | 0.85 |
| **Random Forest (selected)** | **0.91** | **0.85** | **0.88** | 0.99 |
| XGBoost | 0.44 | 0.95 | 0.60 | 0.998 |

**Random Forest was selected as the final model** — it offers the best balance of precision and recall for a practical fraud system (minimizing false alarms while still catching most fraud). XGBoost has the highest raw ROC-AUC but over-flags at its default threshold; tuning its decision threshold is a natural next step.

### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| `amt` | 0.597 |
| `hour_of_day` | 0.210 |
| `category` | 0.111 |
| `age` | 0.031 |
| `city_pop` | 0.025 |
| `distance_miles` | 0.020 |
| `gender` | 0.006 |

This independently confirms the SQL findings — transaction amount and time of day drive fraud predictions, while distance and gender contribute almost nothing, matching the earlier "honest negative results."

## 5. Dashboard

Three-page Power BI dashboard: Overview (KPIs, hour/category fraud rates), Deep Dive (amount, age, gender patterns), and Model Performance (comparison table, feature importance).

![Dashboard Page 1](./page1_overview.png)
![Dashboard Page 2](./page2_deepdive.png)
![Dashboard Page 3](./page3_modelperformance.png)

---

## Key Takeaways

- Fraud in this dataset is driven primarily by **transaction amount** and **time of day** — not by customer-merchant distance or gender, which is confirmed independently by both SQL analysis and ML feature importance.
- A tuned Random Forest classifier catches 85% of fraud while keeping false alarms low (91% precision) — a practical tradeoff for a real fraud review workflow.
- Next steps: threshold tuning for XGBoost, testing on the separate `fraudTest.csv` holdout set, and adding a GenAI layer to auto-generate plain-English explanations for flagged transactions.

## Repository Structure

```
fraud-risk-pipeline/
├── fraud_queries.sql
├── eda.py
├── model.py
├── fraud_by_hour.png
├── fraud_by_category.png
├── amount_distribution.png
├── page1_overview.png
├── page2_deepdive.png
├── page3_modelperformance.png
└── README.md
```
