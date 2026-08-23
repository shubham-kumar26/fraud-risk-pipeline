import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

engine = create_engine('postgresql://postgres:1234@localhost:5432/Sparkov')

query = "SELECT * FROM transactions"
df = pd.read_sql(query, engine)

# Feature 1: Hour of day
df['hour_of_day'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour

# Feature 2: Age
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = (pd.Timestamp('today') - df['dob']).dt.days // 365

# Feature 3: Distance (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 3959 * c

df['distance_miles'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

# Feature 4: Encode category and gender
df['category_encoded'] = df['category'].astype('category').cat.codes
df['gender_encoded'] = df['gender'].map({'M': 0, 'F': 1})

# Select features and target
features = ['hour_of_day', 'age', 'distance_miles', 'category_encoded', 'gender_encoded', 'amt', 'city_pop']
X = df[features]
y = df['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train baseline model
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

print("Model trained successfully")
print("Train accuracy:", model.score(X_train_scaled, y_train))
print("Test accuracy:", model.score(X_test_scaled, y_test))



from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test_scaled)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))




from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)  # Note: tree models don't need scaled data

y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))




from xgboost import XGBClassifier

# Calculate scale_pos_weight for imbalance (ratio of negative to positive class)
scale_pos_weight = (y_train == False).sum() / (y_train == True).sum()

xgb_model = XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("\n--- XGBoost ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))


from sklearn.metrics import roc_auc_score

# ROC-AUC scores for all three models
y_prob_lr = model.predict_proba(X_test_scaled)[:, 1]
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n--- ROC-AUC Scores ---")
print("Logistic Regression:", roc_auc_score(y_test, y_prob_lr))
print("Random Forest:", roc_auc_score(y_test, y_prob_rf))
print("XGBoost:", roc_auc_score(y_test, y_prob_xgb))

# Feature importance from Random Forest
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Feature Importance (Random Forest) ---")
print(importances)

