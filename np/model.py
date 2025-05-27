import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Airline_Delay_Cause.csv")

# Create binary target: 1 if delay > 10, else 0
df['delay_status'] = (df['arr_del15'] > 10).astype(int)

# Drop irrelevant columns and keep only a few weak features
columns_to_keep = ['carrier_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct']  # Exclude weather_ct, etc.
df = df[columns_to_keep + ['delay_status']]

# Drop rows with missing values instead of imputing
df.dropna(inplace=True)

# Prepare features and target
X = df.drop(columns=['delay_status'])
y = df['delay_status']

# Use a smaller training set to reduce performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train logistic regression without standardization
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Reduced Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
