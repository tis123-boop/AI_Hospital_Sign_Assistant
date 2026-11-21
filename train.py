import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("asl_landmarks.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Precision (macro = average across all classes)
precision = precision_score(y_test, y_pred, average='macro')
print("Precision:", precision)

# Detailed Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
# Save model
joblib.dump(model, "asl_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model Saved Successfully")