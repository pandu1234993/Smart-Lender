import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/loan_data.csv")

# Convert required text columns to numbers
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"].astype(str))
df["Married"] = le.fit_transform(df["Married"].astype(str))
df["Credit_History"] = df["Credit_History"].fillna(0)

# Select only required features
X = df[[
    "Gender",
    "Married",
    "ApplicantIncome",
    "LoanAmount",
    "Credit_History"
]]

# Target
y = LabelEncoder().fit_transform(df["Loan_Status"])

# Handle missing values
X = X.fillna(0)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")