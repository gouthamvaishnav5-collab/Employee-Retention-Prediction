# main_xgb.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from joblib import dump

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("aug_train.csv")

# DEBUG: show columns once
print("Columns in dataset:")
print(df.columns)

# -----------------------------
# Target column (CORRECT)
# -----------------------------
TARGET = "target"   # ✅ NOT Attrition

X = df.drop(columns=[TARGET])
y = df[TARGET]

# -----------------------------
# Encode categorical columns
# -----------------------------
encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# XGBoost model
# -----------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# -----------------------------
# Save
# -----------------------------
dump(model, "xgb_model.joblib")
dump(encoders, "encoders.joblib")

print("✅ Model and encoders saved")
