import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv('milk_quality_data.csv')

# Encode
grade_mapping = {'low': 0, 'medium': 1, 'high': 2}
df['grade'] = df['grade'].map(grade_mapping)

# Split
X = df.drop('grade', axis=1)
y = df['grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train XGBoost (Optimized Parameters)
xgb_tuned = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    colsample_bytree=0.5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_tuned.fit(X_train, y_train)

# Evaluate
tuned_pred = xgb_tuned.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, tuned_pred)}")

# Save Model
model_filename = 'milk_quality_xgb_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(xgb_tuned, file)

print(f"Model saved to {model_filename}")
