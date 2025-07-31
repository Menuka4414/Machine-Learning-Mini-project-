import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv(r"C:\Users\menuk\Downloads\HR Churn\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Remove non-informative columns
df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1)

# Encode target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categorical columns
label_cols = df.select_dtypes(include='object').columns
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split data
X = df.drop("Attrition", axis=1)
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model + encoders + column names
joblib.dump(model, "hr_churn_model.pkl")
joblib.dump(encoders, "hr_encoder.pkl")
joblib.dump(X.columns.tolist(), "hr_columns.pkl")

print("Model training complete. Files saved!")
