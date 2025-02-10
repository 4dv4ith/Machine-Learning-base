# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset (assuming it's in CSV format, and you have downloaded it)
url = 'dataset/heart.csv'
df = pd.read_csv(url)

# Preprocessing the data
# For simplicity, assuming 'target' is the column indicating heart disease presence
X = df.drop(columns=['target'])
y = df['target']

# Handle missing values if any (for simplicity, let's drop rows with missing values)
X = X.dropna()
y = y[X.index]

# Scale the features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Train Support Vector Machine classifier
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)

# Evaluate the models

# Decision Tree Evaluation
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions)
dt_conf_matrix = confusion_matrix(y_test, dt_predictions)

# SVM Evaluation
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions)
svm_conf_matrix = confusion_matrix(y_test, svm_predictions)

# Print results
print("Decision Tree Performance:")
print(f"Accuracy: {dt_accuracy}")
print(f"Precision: {dt_precision}")
print(f"Recall: {dt_recall}")
print(f"F1-Score: {dt_f1}")
print(f"Confusion Matrix:\n{dt_conf_matrix}\n")

print("SVM Performance:")
print(f"Accuracy: {svm_accuracy}")
print(f"Precision: {svm_precision}")
print(f"Recall: {svm_recall}")
print(f"F1-Score: {svm_f1}")
print(f"Confusion Matrix:\n{svm_conf_matrix}")
