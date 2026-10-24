import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the public student performance dataset
url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

import io
import zipfile
import urllib.request

response = urllib.request.urlopen(url)
with zipfile.ZipFile(io.BytesIO(response.read())) as outer_zip:
    inner_zip_data = outer_zip.read("student.zip")

    with zipfile.ZipFile(io.BytesIO(inner_zip_data)) as inner_zip:
        csv_file = next(
            name for name in inner_zip.namelist()
            if name.endswith("student-mat.csv")
        )
        with inner_zip.open(csv_file) as file:
            df = pd.read_csv(file, sep=";")

print("Dataset loaded successfully!")
print(df.head())

# Select input features
features = ["studytime", "absences", "G1", "G2", "failures"]
X = df[features]

# Define the target: 1 = Pass, 0 = Needs Support
y = (df["G3"] >= 10).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train the machine-learning model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate the model on unseen test data
predictions = model.predict(X_test)

print("Model training completed!")
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model and its feature names
joblib.dump(
    {
        "model": model,
        "features": features
    },
    "student_model.pkl"
)

print("\nSaved model as student_model.pkl")